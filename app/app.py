import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
from io import BytesIO
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# ==========================================================
#                PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="RetinaX • AI OCT Analyzer",
    page_icon="🧠",
    layout="wide",
)

# ==========================================================
#                STYLES (clean professional)
# ==========================================================
st.markdown(
    """
    <style>
    :root {
      --card-bg: #FFFFFF;
      --page-bg: #F6F8FA;
      --muted: #6B7280;
      --accent: #1D4ED8;
      --danger: #B91C1C;
      --bar-bg: #E6EEF8;
      --bar-fill: linear-gradient(90deg,#1D4ED8,#2563EB);
    }

    body { background: var(--page-bg); font-family: "Segoe UI", Roboto, Arial; }

    .glass {
      background: var(--card-bg);
      border-radius: 14px;
      padding: 18px;
      border: 1px solid rgba(15,23,42,0.06);
      box-shadow: 0 8px 24px rgba(15,23,42,0.06);
      margin-bottom: 18px;
    }

    .main-title {
      font-size: 36px;
      font-weight: 800;
      color: #0F172A;
      text-align: center;
      margin-bottom: 6px;
    }
    .main-sub { text-align:center; color:var(--muted); margin-bottom:18px; }

    .section-title{ font-weight:700; font-size:16px; color:#0F172A; margin-bottom:6px; }
    .section-caption { color:var(--muted); font-size:13px; margin-bottom:10px; }

    .pred-badge {
      display:inline-block;
      padding:6px 12px;
      border-radius:999px;
      border:1px solid rgba(185,28,28,0.14);
      color: var(--danger);
      background: rgba(249,250,251,0.9);
      font-weight:700;
    }

    /* custom bar component */
    .conf-row { margin: 6px 0; display:flex; align-items:center; gap:12px; }
    .conf-label { width:160px; font-weight:600; color:#0F172A; }
    .conf-bar { flex:1; background:var(--bar-bg); height:16px; border-radius:999px; position:relative; overflow:hidden; }
    .conf-bar-fill { height:100%; border-radius:999px; background: var(--bar-fill); position:absolute; left:0; top:0; }
    .conf-pct { width:64px; text-align:right; font-weight:700; color:#0F172A; }

    .tiny-pill {
      display:inline-block;
      padding:6px 10px;
      border-radius:999px;
      background:#F3F4F6;
      color: #6B7280;
      border:1px solid rgba(15,23,42,0.04);
      font-weight:600;
      width: fit-content;
    }

    /* images area spacing */
    .images-row { display:flex; gap:28px; align-items:flex-start; margin-top:18px; }
    .img-card { background:#FFFFFF; border-radius:10px; padding:8px; border:1px solid rgba(15,23,42,0.04); }

    /* small muted */
    .small-muted { color:var(--muted); font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# small helper to render the confidence bars as HTML (no empty large bars)
def render_confidence_html(class_names, probs):
    rows = []
    for cname, p in zip(class_names, probs):
        pct = float(p) * 100.0
        if pct < 0.5:
            # render small pill instead of empty bar
            rows.append(
                f'<div class="conf-row"><div class="conf-label">{cname}</div>'
                f'<div class="tiny-pill">{pct:.2f}%</div></div>'
            )
        else:
            # clamp width to 100%
            width_pct = min(100.0, pct)
            # fill gradient + text
            rows.append(
                '<div class="conf-row">'
                f'<div class="conf-label">{cname}</div>'
                f'<div class="conf-bar"><div class="conf-bar-fill" style="width:{width_pct:.2f}%;"></div></div>'
                f'<div class="conf-pct">{pct:.2f}%</div>'
                '</div>'
            )
    return "<br>".join(rows)


# ==========================================================
#                MODEL LOADING
# ==========================================================
@st.cache_resource
def load_model():
    path = "models/mobilenetv2_retinal_oct_best_20251201_235524.pth"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


model, CLASS_NAMES = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ==========================================================
#                GRAD-CAM
# ==========================================================
def generate_gradcam(model, image_tensor):
    model.eval()
    target_layer = model.features[-1]

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    preds = model(image_tensor)
    pred_class = preds.argmax(dim=1)

    model.zero_grad()
    preds[0, pred_class].backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=[2, 3], keepdim=True)
    gradcam_map = (weights * act).sum(dim=1).squeeze()

    gradcam_map = torch.relu(gradcam_map)
    if gradcam_map.max() != 0:
        gradcam_map = gradcam_map / gradcam_map.max()
    gradcam_map = gradcam_map.cpu().detach().numpy()
    gradcam_map = cv2.resize(gradcam_map, (224, 224))
    gradcam_map = np.uint8(255 * gradcam_map)

    heatmap = cv2.applyColorMap(gradcam_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # remove hooks
    fh.remove(); bh.remove()
    return heatmap

# ==========================================================
#              PDF REPORT GENERATOR (fixed spacing, safe images)
# ==========================================================
def _draw_watermark(c, text="RETINAX AI"):
    width, height = letter
    c.saveState()
    c.setFont("Helvetica-Bold", 60)
    c.setFillColorRGB(0.93, 0.95, 0.98)
    c.translate(width / 2, height / 2)
    c.rotate(30)
    c.drawCentredString(0, 0, text)
    c.restoreState()


def wrap_text(text, width=90):
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            lines.extend(textwrap.wrap(paragraph, width))
        lines.append("")
    return lines


def generate_pdf(pred_label, probs, orig_img=None, heatmap=None):
    """
    Produce a 2-page PDF with:
    - top summary + neat bars (fixed spacing)
    - suggestion box
    - heatmap card lower on the page with images that never overlap
    - page 2: clinical notes + signature
    """
    file_path = "retina_diagnosis_report.pdf"
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    _draw_watermark(c)

    # header band
    c.setFillColor(colors.HexColor("#1D4ED8"))
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "RetinaX OCT Diagnostic Report")

    # primary detection box
    c.setFillColor(colors.HexColor("#F5F7FB"))
    c.roundRect(40, height - 200, width - 80, 100, 12, fill=True, stroke=False)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, height - 150, "Primary Detection")
    c.setFillColor(colors.HexColor("#B91C1C"))
    c.setFont("Helvetica-Bold", 22)
    c.drawString(60, height - 180, pred_label.upper())

    # ---- confidence bars (fixed rows) ----
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(50, height - 240, "Confidence Breakdown")

    # layout constants
    bars_start_y = height - 270
    bar_x = 60
    bar_width = 420
    bar_height = 14
    bar_gap = 34

    for i, cname in enumerate(CLASS_NAMES):
        p = float(probs[i])
        pct = p * 100.0
        y = bars_start_y - i * bar_gap
        # label
        c.setFont("Helvetica", 11)
        c.setFillColor(colors.black)
        c.drawString(bar_x, y + 2, f"{cname}")
        # small pill for tiny probs
        if pct < 0.5:
            c.setFillColor(colors.HexColor("#F3F4F6"))
            c.roundRect(bar_x + 130, y - 10, 60, bar_height, 8, fill=True, stroke=False)
            c.setFillColor(colors.HexColor("#6B7280"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(bar_x + 140, y + 2, f"{pct:.2f}%")
        else:
            # background
            c.setFillColor(colors.HexColor("#E6EEF8"))
            c.roundRect(bar_x + 130, y - 10, bar_width, bar_height, 8, fill=True, stroke=False)
            # foreground
            fg_w = min(bar_width, bar_width * p)
            c.setFillColor(colors.HexColor("#1D4ED8"))
            c.roundRect(bar_x + 130, y - 10, fg_w, bar_height, 8, fill=True, stroke=False)
            # percent on right of bar
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica-Bold", 10)
            c.drawRightString(bar_x + 130 + bar_width, y + 2, f"{pct:.2f}%")

    # compute bottom of bars region
    bars_bottom = bars_start_y - (len(CLASS_NAMES) - 1) * bar_gap - 16

    # suggestion box below bars
    sugg_height = 72
    sugg_top = bars_bottom - 24
    c.setFillColor(colors.HexColor("#FFF7E6"))
    c.roundRect(40, sugg_top - sugg_height, width - 80, sugg_height, 8, fill=True, stroke=False)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(60, sugg_top - 30, "Suggested Action:")
    suggestions = {
        "AMD": "Regular OCT scans advised. Consider anti-VEGF evaluation.",
        "CNV": "Immediate ophthalmologist consultation recommended.",
        "CSR": "Reduce stress & steroid usage if applicable.",
        "DME": "Monitor glucose closely. Schedule diabetic eye exam.",
        "DR": "Comprehensive retinal exam needed within 30 days.",
        "DRUSEN": "Risk monitoring recommended every 6–12 months.",
        "MH": "Urgent retinal specialist visit required.",
        "NORMAL": "No abnormalities detected. Routine exam is enough."
    }
    c.setFont("Helvetica", 11)
    c.drawString(60, sugg_top - 50, suggestions.get(pred_label, "Follow-up recommended."))

    # ---- Heatmap card (fixed area near bottom with enough height) ----
    heat_card_h = 240
    heat_card_y = 110
    c.setFillColor(colors.HexColor("#F5F7FB"))
    c.roundRect(40, heat_card_y, width - 80, heat_card_h, 12, fill=True, stroke=False)

    # Title
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, heat_card_y + heat_card_h - 26, "AI Focus Map (Grad-CAM)")

    # explanation (wrapped)
    c.setFont("Helvetica", 11)
    expl = ("The heatmap highlights regions of the scan that influenced the AI. "
            "Red/yellow indicate stronger influence; blue indicates lower influence.")
    lines = textwrap.wrap(expl, 86)
    tx = 60
    ty = heat_card_y + heat_card_h - 46
    for line in lines:
        c.drawString(tx, ty, line)
        ty -= 14

    # images area inside heat card (ensure they fit and don't overlap)
    if orig_img is not None and heatmap is not None:
        # overlay a small arrow on a copy of heatmap for emphasis
        heat_arr = heatmap.copy()
        h, w, _ = heat_arr.shape
        # arrow coords relative to heatmap center-ish
        start = (int(w * 0.18), int(h * 0.18))
        end = (int(w * 0.48), int(h * 0.48))
        cv2.arrowedLine(heat_arr, start, end, (255, 255, 255), 3, tipLength=0.32)

        # resize images to fit in card
        target_w = 220
        target_h = 160
        orig_small = orig_img.resize((target_w, target_h))
        heat_small = Image.fromarray(heat_arr).resize((target_w, target_h))

        buf1 = BytesIO(); orig_small.save(buf1, format="PNG"); buf1.seek(0)
        buf2 = BytesIO(); heat_small.save(buf2, format="PNG"); buf2.seek(0)

        left_x = 60
        right_x = left_x + target_w + 40
        imgs_y = heat_card_y + 18

        # draw images using ImageReader (safe)
        c.drawImage(ImageReader(buf1), left_x, imgs_y, width=target_w, height=target_h, mask='auto')
        c.drawImage(ImageReader(buf2), right_x, imgs_y, width=target_w, height=target_h, mask='auto')

        # labels
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)
        c.drawString(left_x, imgs_y - 12, "Original OCT Scan")
        c.drawString(right_x, imgs_y - 12, "AI Attention Heatmap")

    # footer page 1
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(50, 40, "AI system for clinical decision support. Not a standalone diagnosis.")
    c.drawString(50, 25, "RetinaX Deep Vision Diagnostics © 2025")

    # ---------- PAGE 2: Clinical notes ----------
    c.showPage()
    _draw_watermark(c)
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawString(50, height - 70, "Clinical Interpretation & Notes")

    explanations = {
        "AMD": ("Age-related Macular Degeneration (AMD) involves progressive damage to the macula, "
                "leading to central vision loss. Early detection and treatment can slow progression."),
        "CNV": ("Choroidal Neovascularization (CNV) causes abnormal blood vessels under the retina "
                "and often needs urgent treatment."),
        "CSR": ("Central Serous Retinopathy (CSR) is fluid accumulation under the retina; many cases "
                "resolve but chronic cases require monitoring."),
        "DME": ("Diabetic Macular Edema (DME) results from vascular leakage causing macular swelling; "
                "control diabetes and consult retina specialist."),
        "DR": ("Diabetic Retinopathy (DR) arises from chronic diabetes damaging retinal vessels; "
               "regular screening and treatment are crucial."),
        "DRUSEN": ("Drusen are subretinal deposits associated with AMD risk; monitor and manage risks."),
        "MH": ("Macular Hole is a structural break in the macula — surgical repair may be needed."),
        "NORMAL": ("No major structural abnormalities detected. Continue routine screening as advised.")
    }
    exp_text = explanations.get(pred_label, "No additional information available for this category.")
    lines = wrap_text(exp_text, width=95)
    text_obj = c.beginText()
    text_obj.setTextOrigin(50, height - 120)
    text_obj.setFont("Helvetica", 12)
    for ln in lines:
        text_obj.textLine(ln)
    c.drawText(text_obj)

    # suggestion box on page 2
    c.setFillColor(colors.HexColor("#FFF7E6"))
    c.roundRect(50, 200, width - 100, 88, 10, fill=True, stroke=False)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(64, 238, "Suggested Action:")
    c.setFont("Helvetica", 11)
    c.drawString(64, 220, suggestions.get(pred_label, "Follow-up recommended."))

    # signature
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, 120, "Reviewed by:")
    c.setFont("Helvetica-Oblique", 15)
    c.drawString(50, 100, "RetinaX AI Assist")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawString(50, 85, "Computer-generated report to support clinical decision making.")

    # stamp
    c.setFillColor(colors.HexColor("#1D4ED8"))
    c.rect(width - 180, 70, 110, 60, fill=False, stroke=True)
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawString(width - 172, 110, "RETINAX AI")
    c.setFont("Helvetica", 8)
    c.drawString(width - 172, 95, "Autogenerated")
    c.drawString(width - 172, 82, "Diagnostic Support")

    # footer page 2
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(50, 40, "Final diagnosis and treatment decisions must be made by a qualified ophthalmologist.")
    c.drawString(50, 25, "RetinaX Deep Vision Diagnostics © 2025")

    c.save()
    return file_path


# ==========================================================
#              IMAGE PREPROCESSING & PREDICTION
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def predict(image):
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, 1)[0].cpu().numpy()
    label = CLASS_NAMES[np.argmax(probs)]
    heatmap = generate_gradcam(model, img_t)
    return label, probs, heatmap


# ==========================================================
#                     MAIN UI
# ==========================================================
st.markdown("<div class='main-title'>RetinaX • AI Retinal Disease Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='main-sub'>Quick, professional OCT screening with downloadable clinical-style report</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📤 Upload OCT Scan</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Upload a macular B-scan (PNG / JPG). Best results for centered macular scans.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(" ", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    img = None
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded OCT Scan", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧠 AI Diagnosis</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>MobileNetV2 backbone, fine-tuned on multi-class retinal OCT dataset.</div>", unsafe_allow_html=True)

    if img is not None:
        with st.spinner("Analyzing scan with RetinaX AI..."):
            label, probs, heatmap = predict(img)

        st.markdown(f"<span class='pred-badge'>PREDICTION: {label.upper()}</span>", unsafe_allow_html=True)
        st.write("")
        st.write("#### Confidence Scores")
        html_bars = render_confidence_html(CLASS_NAMES, probs)
        st.markdown(html_bars, unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Tiny probabilities (under 0.5%) are shown as subdued pills to avoid noisy empty bars.</div>", unsafe_allow_html=True)

        # produce PDF and download
        pdf_path = generate_pdf(label, probs, img, heatmap)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Diagnostic Report (PDF)", f, file_name="RetinaX_Report.pdf")
    else:
        st.info("Upload an OCT scan on the left to run RetinaX AI.")
    st.markdown("</div>", unsafe_allow_html=True)

# Grad-CAM visualization
if img is not None:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Grad-CAM Attention Map", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Visual explanation of regions which influenced the prediction.</div>", unsafe_allow_html=True)
    colA, colB = st.columns([1, 1])
    with colA:
        st.image(img, caption="Original OCT Image", use_container_width=True)
    with colB:
        st.image(heatmap, caption="AI Attention Heatmap", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("🧠 RetinaX System Info")
st.sidebar.write(f"Device: **{device.upper()}**")
st.sidebar.write("Backbone: **MobileNetV2**")
st.sidebar.write("Framework: **PyTorch + Streamlit**")
st.sidebar.write("Report Engine: **ReportLab PDF**")
st.sidebar.markdown("---")
st.sidebar.write("Made by **Puneeth B J** • For clinical decision support only.")

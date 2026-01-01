🧠 Retinal OCT Disease Classification – 8-Class Medical AI (C8 Dataset)


A deep-learning powered medical imaging system that classifies 8 retinal diseases using Optical Coherence Tomography (OCT) scans — with a clean Streamlit GUI, MobileNetV2 transfer-learning, and Grad-CAM heatmaps for explainability.

🩺 Supported Retinal Conditions (8-Class C8 Dataset)
Class	Condition
1️⃣	AMD – Age-related Macular Degeneration
2️⃣	CNV – Choroidal Neovascularization
3️⃣	CSR – Central Serous Retinopathy
4️⃣	DME – Diabetic Macular Edema
5️⃣	DR – Diabetic Retinopathy
6️⃣	DRUSEN
7️⃣	MH – Macular Hole
8️⃣	NORMAL

📦 Dataset Source (Kaggle)
🔗 Retinal OCT C8 dataset → https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8

🚀 Tech Stack
Component	Choice
CNN Backbone	MobileNetV2 (EfficientNetB0 optional)
Framework	PyTorch
GUI	Streamlit
Explainability	Grad-CAM Heatmaps
Hardware	Trained locally on GPU
🏗️ Project Features

✔ Deep Learning – Transfer Learning (MobileNetV2)
✔ Heatmap visualization for ROI (Grad-CAM)
✔ Streamlit Web-App to upload OCT scans
✔ Shows prediction + probability scores
✔ Handles high-resolution OCT images
✔ Clean architecture & modular pipeline

🧬 Model Pipeline

A[OCT Image Upload] --> B[Preprocessing & Resize 224x224]
B --> C[MobileNetV2 Transfer Learning]
C --> D[Softmax Classification – 8 Classes]
D --> E[Grad-CAM Heatmap Overlay]
E --> F[Streamlit UI Output]

🧑‍💻 Installation & Local Run
1️⃣ Clone Repo
git clone https://github.com/Pandapuneeth/retinal_disease_detector.git
cd retinal_disease_detector

2️⃣ Create Environment
conda create -n retina python=3.10
conda activate retina
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app/app.py

🤝 Contributing

Feel free to fork, improve, and submit PRs — especially UI / deployment upgrades.

🧾 License

MIT License — free for academic & commercial use.

🌟 Show Some Love

If this project helped you — ⭐ star the repo!

💬 Author

👤 Puneeth B J
AI/ML Engineer — Computer Vision • Medical AI • Cybersecurity
🔗 LinkedIn: www.linkedin.com/in/puneeth-b-j-037bba252
🔗 GitHub: Pandapuneeth

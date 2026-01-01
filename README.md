# Retinal OCT Disease Classification (8 Classes)

Deep learning model that classifies 8 retinal diseases using Optical Coherence Tomography (OCT) scans.  
Includes a Streamlit web app UI, MobileNetV2 transfer learning, and Grad-CAM visualization.

---

## Supported Classes
- AMD – Age-related Macular Degeneration
- CNV – Choroidal Neovascularization
- CSR – Central Serous Retinopathy
- DME – Diabetic Macular Edema
- DR – Diabetic Retinopathy
- DRUSEN
- MH – Macular Hole
- NORMAL – Healthy Retina

---

## Dataset
Kaggle dataset:
https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8

---

## Tech Stack
- Framework: PyTorch
- Model: MobileNetV2 (transfer learning)
- Interface: Streamlit
- Explainability: Grad-CAM heatmaps
- Training script: train.py
- App: app/app.py

---

## Features
- Trainable PyTorch model
- Upload image → classify disease
- Shows probability score
- Grad-CAM heatmap highlight
- Organized modular project structure

---

## How To Run (Local)

### Clone repository
git clone https://github.com/Pandapuneeth/retinal_disease_detector.git
cd retinal_disease_detector
Create environment and install dependencies

conda create -n retina python=3.10
conda activate retina
pip install -r requirements.txt
Run the Streamlit app

streamlit run app/app.py
Training (Optional)
To retrain the model:


python train.py
Model (.pth) gets saved in /models.

Contributing
Pull requests and improvements are welcome.

License
MIT License

Author
Puneeth B J
LinkedIn: https://www.linkedin.com/in/puneeth-b-j-037bba252
GitHub: https://github.com/Pandapuneeth
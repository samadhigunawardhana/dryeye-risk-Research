# Computer Vision-Based Algorithm for Detecting Dry Eye Risk Using Webcam Blink Dynamics

**Coventry Index:** 14945984  
**NIBM Index:** YR4COBSCCOMP232P-017  
**Registered Name:** **M.D.S.G.K Gunawardhana**

---

## Overview

This project implements an end-to-end computer-vision pipeline to estimate **Dry Eye risk** from short eye videos (5–20 s).  
It extracts blink dynamics and a scleral redness index, then applies a trained multi-class classifier to output **Low / Medium / High** risk.  
A Streamlit dashboard provides a simple UI with a **7-item OSDI questionnaire**.

**What you can do**
1. Upload a short eye video → get features + risk label.  
2. (Optional) Enter OSDI answers → the app computes the OSDI score and shows it with the prediction.  
3. Batch-predict from a CSV of features.  
4. Re-train the model from `dataset.csv` using a single script.

---

## Files (Runtime + Training)

app_dashboard_best.py # Streamlit dashboard (main app)
extract_features.py # Video → features (blink + redness)
best_model.joblib # Trained classifier (multi-class)
label_encoder.joblib # Label encoder for class names
train_model_best.py # Re-train from dataset.csv (macro-F1 model selection)
dataset.csv # Training data (features + label)
metrics.json # Training metrics summary (JSON)
confusion_matrix.png # Holdout confusion matrix
requirements.txt # Python dependencies
recordings/ # (optional) small demo videos for testing


> **Tip:** keep only one small demo video in `recordings/` to keep the submission size small.

---

## Installation

> Use **Python 3.9–3.10** in a virtual environment.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


Run the Dashboard
# from the virtual environment
python -m streamlit run app_dashboard_best.py

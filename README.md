# 🏥 AI-Based Multi-Disease Risk Level Assessment System

> **AI-powered preventive healthcare tool assessing risk for 5 diseases using ML probability scores.**

---

## 🚀 Quick Start

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd health_app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train ML models (first time only)
```bash
python train_models.py
```

### 4. Launch the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
health_app/
│
├── app.py                     ← 🚀 Main Streamlit application
├── train_models.py            ← 🤖 ML training script (all 5 diseases)
├── requirements.txt           ← 📦 Python dependencies
├── README.md                  ← 📖 This file
│
├── utils/
│   ├── __init__.py
│   ├── risk_assessment.py     ← ⚠️ Risk levels, health score, tips
│   └── preprocessing.py      ← 🔄 Model loading, prediction, XAI
│
├── models/                    ← 💾 Auto-generated after training
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── parkinsons_model.pkl
│   ├── ckd_model.pkl
│   ├── liver_model.pkl
│   └── all_metrics.json
│
└── data/                      ← 📊 Sample CSVs + format guide
    ├── DATASET_FORMAT.md
    ├── diabetes_sample.csv
    ├── heart_sample.csv
    ├── parkinsons_sample.csv
    ├── ckd_sample.csv
    └── liver_sample.csv
```

---

## 🧠 Disease Models

| Disease | Algorithm | Key Features |
|---------|-----------|--------------|
| Diabetes | Random Forest | Glucose, BMI, Insulin, Age, DPF |
| Heart Disease | Random Forest | Chest pain, Cholesterol, HR, ECG |
| Parkinson's | Random Forest | Jitter, Shimmer, HNR, RPDE, PPE |
| Kidney (CKD) | Random Forest | Creatinine, Urea, Haemoglobin |
| Liver Disease | Random Forest | Bilirubin, ALT, AST, Albumin |

---

## ⚡ Innovation Features

### 🔀 Cross-Disease Risk Amplifier
- High Diabetes risk → slightly increases Heart Disease probability
- High Heart Disease risk → slightly increases CKD probability  
- High CKD risk → slightly increases Liver Disease probability

### 👤 Personalized Age-Adjusted Thresholds
| Age Group | Low Risk | Medium Risk | High Risk |
|-----------|----------|-------------|-----------|
| Child (<18) | 0–25% | 26–55% | >55% |
| Young (18-35) | 0–35% | 36–68% | >68% |
| Adult (36-55) | 0–33% | 34–66% | >66% ← Default |
| Senior (56-70) | 0–28% | 29–60% | >60% |
| Elderly (71+) | 0–22% | 23–52% | >52% |

### 🧠 Explainable AI (XAI)
- RandomForest feature importance scores
- Top 8 contributing biomarkers displayed
- Helps users understand prediction rationale

### 📈 Risk Progression Simulation
- 5-year projected risk trend
- Based on untreated logistic growth model
- Visualized as interactive line chart

### 🏆 Overall Health Score
- Weighted composite of all disease probabilities
- Score 0–100 (higher = healthier)
- Grades: Excellent / Good / Fair / Poor / Critical

---

## ☁️ Streamlit Cloud Deployment

### Step 1: Create GitHub repository
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path**: `app.py`
5. Click **Deploy**

### Step 3: Add `.streamlit/config.toml` for theming
```toml
[theme]
primaryColor = "#6c63ff"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1d27"
textColor = "#e8eaf0"
```

> **Note**: Models are auto-trained on first launch (takes ~60 seconds).  
> Add `models/` folder to the repo to skip training on deployment.

---

## 🖥️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                USER INTERFACE LAYER                  │
│   Streamlit Web App | Dark Mode | Step Wizard        │
└─────────────────────────┬───────────────────────────┘
                          │ User Input
┌─────────────────────────▼───────────────────────────┐
│               DATA PROCESSING LAYER                  │
│   Input Validation | Normalization | Feature Eng.    │
└─────────────────────────┬───────────────────────────┘
                          │ Processed Features
┌─────────────────────────▼───────────────────────────┐
│              MACHINE LEARNING LAYER                  │
│   Random Forest | StandardScaler | Pipeline          │
│   predict_proba() → probability [0, 1]               │
└─────────────────────────┬───────────────────────────┘
                          │ Raw Probabilities
┌─────────────────────────▼───────────────────────────┐
│             RISK ASSESSMENT LAYER                    │
│   Age-Threshold Adjustment | Cross-Disease Amplifier │
│   Health Score | Risk Levels | Progression Sim       │
└─────────────────────────┬───────────────────────────┘
                          │ Risk Results
┌─────────────────────────▼───────────────────────────┐
│            OUTPUT PRESENTATION LAYER                 │
│   Risk Cards | Gauge Charts | XAI | Tips | Score     │
└─────────────────────────────────────────────────────┘
```

---

## 📊 ML Metrics Summary (Synthetic Data)

| Disease | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Diabetes | ~0.88 | ~0.87 | ~0.88 | ~0.87 |
| Heart Disease | ~0.89 | ~0.88 | ~0.89 | ~0.88 |
| Parkinson's | ~0.92 | ~0.91 | ~0.92 | ~0.91 |
| CKD | ~0.90 | ~0.89 | ~0.90 | ~0.89 |
| Liver Disease | ~0.82 | ~0.81 | ~0.82 | ~0.81 |

*Actual values vary with each training run. View live metrics in the app.*

---

## 🔮 Future Extensions

1. **More Diseases**: Add new disease by creating `generate_*_data()` + registering in `train_models.py`
2. **Real Datasets**: Replace synthetic generators with real clinical datasets
3. **EHR Integration**: Connect to FHIR-compliant health records API
4. **Mobile App**: Wrap with Streamlit mobile or convert to React Native
5. **Federated Learning**: Privacy-preserving distributed training concept
6. **Voice Upload**: Real PRAAT feature extraction from .wav files
7. **Longitudinal Tracking**: Store assessment history per user (with consent)

---

## ⚕️ Disclaimer

This application is for **educational and informational purposes only**. It uses ML models trained on synthetic data. Results should **not** be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider.

---

*Built with ❤️ using Python, Streamlit, scikit-learn, and Plotly*

# Sample Dataset Format Guide

## Overview
This document describes the expected column formats for each disease dataset.
The `train_models.py` script auto-generates synthetic data, but real datasets can be substituted.

---

## 1. Diabetes Dataset (`data/diabetes_sample.csv`)
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Pregnancies | int | 0–17 | Number of pregnancies |
| Glucose | float | 50–250 | Plasma glucose (mg/dL) |
| BloodPressure | float | 40–130 | Diastolic BP (mmHg) |
| SkinThickness | float | 0–70 | Triceps skinfold (mm) |
| Insulin | float | 0–500 | 2-hour serum insulin (μU/mL) |
| BMI | float | 15–65 | Body mass index |
| DiabetesPedigreeFunction | float | 0.08–2.5 | Genetic score |
| Age | float | 21–81 | Age in years |
| **Outcome** | int | 0 or 1 | **Target: 1=Diabetic** |

**Real Dataset**: PIMA Indian Diabetes Dataset (UCI / Kaggle)

---

## 2. Heart Disease Dataset (`data/heart_sample.csv`)
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| age | float | 29–77 | Age in years |
| sex | int | 0=F, 1=M | Biological sex |
| cp | int | 0–3 | Chest pain type |
| trestbps | float | 90–200 | Resting blood pressure |
| chol | float | 120–600 | Cholesterol (mg/dL) |
| fbs | int | 0 or 1 | Fasting blood sugar >120 |
| restecg | int | 0–2 | Resting ECG |
| thalach | float | 70–210 | Max heart rate |
| exang | int | 0 or 1 | Exercise angina |
| oldpeak | float | 0–6 | ST depression |
| slope | int | 0–2 | ST slope |
| ca | int | 0–4 | Major vessels |
| thal | int | 0–3 | Thalassemia type |
| **target** | int | 0 or 1 | **Target: 1=Disease** |

**Real Dataset**: UCI Heart Disease Dataset (Cleveland)

---

## 3. Parkinson's Dataset (`data/parkinsons_sample.csv`)
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| MDVP:Fo(Hz) | float | 80–260 | Avg vocal frequency |
| MDVP:Fhi(Hz) | float | 100–600 | Max vocal frequency |
| MDVP:Flo(Hz) | float | 65–240 | Min vocal frequency |
| MDVP:Jitter(%) | float | 0.001–0.08 | Frequency variation |
| MDVP:Shimmer | float | 0.009–0.2 | Amplitude variation |
| HNR | float | 8–34 | Harmonics-to-noise ratio (dB) |
| NHR | float | 0.0006–0.3 | Noise-to-harmonics ratio |
| RPDE | float | 0.26–0.69 | Recurrence period density |
| DFA | float | 0.57–0.83 | Detrended fluctuation |
| PPE | float | 0.04–0.53 | Pitch period entropy |
| **status** | int | 0 or 1 | **Target: 1=Parkinson's** |

**Real Dataset**: UCI Parkinson's Voice Dataset

---

## 4. CKD Dataset (`data/ckd_sample.csv`)
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| age | float | 2–90 | Age in years |
| bp | float | 50–180 | Blood pressure |
| sg | float | 1.005–1.025 | Urine specific gravity |
| al | int | 0–4 | Albumin level |
| su | int | 0–4 | Sugar level |
| rbc | int | 0 or 1 | Red blood cells (0=normal) |
| bu | float | 1.5–390 | Blood urea (mg/dL) |
| sc | float | 0.4–76 | Serum creatinine (mg/dL) |
| sod | float | 111–163 | Sodium (mEq/L) |
| pot | float | 2.5–47 | Potassium (mEq/L) |
| hemo | float | 3.1–17.8 | Haemoglobin (g/dL) |
| htn | int | 0 or 1 | Hypertension |
| dm | int | 0 or 1 | Diabetes |
| **classification** | int | 0 or 1 | **Target: 1=CKD** |

**Real Dataset**: UCI Chronic Kidney Disease Dataset

---

## 5. Liver Disease Dataset (`data/liver_sample.csv`)
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Age | float | 4–90 | Age in years |
| Gender | int | 0=F, 1=M | Biological sex |
| Total_Bilirubin | float | 0.4–75 | Total bilirubin (mg/dL) |
| Direct_Bilirubin | float | 0.1–20 | Direct bilirubin (mg/dL) |
| Alkaline_Phosphotase | float | 60–2200 | ALP (IU/L) |
| Alamine_Aminotransferase | float | 10–2000 | ALT/SGPT (IU/L) |
| Aspartate_Aminotransferase | float | 10–5000 | AST/SGOT (IU/L) |
| Total_Protiens | float | 2.7–9.6 | Total proteins (g/dL) |
| Albumin | float | 0.9–5.5 | Albumin (g/dL) |
| Albumin_and_Globulin_Ratio | float | 0.3–2.8 | A/G ratio |
| **Dataset** | int | 0 or 1 | **Target: 1=Liver Disease** |

**Real Dataset**: ILPD (Indian Liver Patient Dataset)

---

## Using Real Datasets
1. Download the dataset from UCI or Kaggle
2. Rename columns to match the format above
3. Replace the `generate_*_data()` function calls in `train_models.py` with `pd.read_csv()`
4. Re-run `python train_models.py` to retrain

```python
# Example: Replace synthetic diabetes data with real data
df_diabetes = pd.read_csv("path/to/pima_diabetes.csv")
# Make sure column names match exactly
```

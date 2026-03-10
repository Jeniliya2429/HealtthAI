"""
=============================================================================
  AI Multi-Disease Risk Assessment System  |  train_models.py  |  v2.0
  Trains 10 disease models: Diabetes, Heart, Parkinson's, CKD, Liver,
  Stroke, Breast Cancer, Thyroid, PCOS, Anemia
=============================================================================
"""

import numpy as np
import pandas as pd
import pickle, os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline

MODELS_DIR   = "models"
DATA_DIR     = "data"
N_SAMPLES    = 3000
RANDOM_STATE = 42
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)


def build_pipeline():
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([('scaler', StandardScaler()), ('classifier', clf)])


def train_save(df, feature_cols, target_col, disease_key, disease_name):
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(int)
    df.to_csv(os.path.join(DATA_DIR, f"{disease_key}_sample.csv"), index=False)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_sc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring='accuracy')
    metrics = {
        "disease"  : disease_name,
        "accuracy" : round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred, average='weighted', zero_division=0), 4),
        "recall"   : round(recall_score(y_te, y_pred, average='weighted', zero_division=0), 4),
        "f1_score" : round(f1_score(y_te, y_pred, average='weighted', zero_division=0), 4),
        "cv_mean"  : round(cv_sc.mean(), 4),
        "cv_std"   : round(cv_sc.std(), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
    }
    print(f"  ✅ {disease_name:<24} Acc={metrics['accuracy']:.4f} "
          f"F1={metrics['f1_score']:.4f} CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")
    with open(os.path.join(MODELS_DIR, f"{disease_key}_model.pkl"), 'wb') as f:
        pickle.dump({'pipeline': pipe, 'feature_cols': feature_cols}, f)
    with open(os.path.join(MODELS_DIR, f"{disease_key}_meta.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ─── DATA GENERATORS ─────────────────────────────────────────────────────────

def gen_diabetes(n=N_SAMPLES):
    rng = np.random.default_rng(RANDOM_STATE)
    preg=rng.integers(0,17,n).astype(float); gluc=rng.normal(120,32,n).clip(50,250)
    bp=rng.normal(72,12,n).clip(40,130); skin=rng.normal(29,15,n).clip(0,70)
    ins=rng.exponential(80,n).clip(0,500); bmi=rng.normal(32,7,n).clip(15,65)
    dpf=rng.exponential(0.47,n).clip(0.08,2.5); age=rng.integers(21,81,n).astype(float)
    lo=-6.0+0.032*gluc+0.012*bmi+0.010*age+0.005*ins+0.8*dpf+0.05*preg
    y=(1/(1+np.exp(-lo))+rng.normal(0,0.04,n)>0.5).astype(int)
    return pd.DataFrame({'Pregnancies':preg,'Glucose':gluc,'BloodPressure':bp,
        'SkinThickness':skin,'Insulin':ins,'BMI':bmi,
        'DiabetesPedigreeFunction':dpf,'Age':age,'Outcome':y})


def gen_heart(n=N_SAMPLES):
    rng=np.random.default_rng(RANDOM_STATE+1)
    age=rng.integers(29,77,n).astype(float); sex=rng.integers(0,2,n).astype(float)
    cp=rng.integers(0,4,n).astype(float); tbp=rng.normal(131,17,n).clip(90,200)
    cho=rng.normal(246,52,n).clip(120,600); fbs=(rng.random(n)<0.15).astype(float)
    rec=rng.integers(0,3,n).astype(float); tha=rng.normal(149,23,n).clip(70,210)
    exa=(rng.random(n)<0.33).astype(float); old=rng.exponential(1.0,n).clip(0,6)
    slo=rng.integers(0,3,n).astype(float); ca=rng.integers(0,5,n).astype(float)
    thl=rng.integers(0,4,n).astype(float)
    lo=-4.5+0.04*age+0.5*sex+0.01*tbp+0.003*cho+0.3*fbs+0.8*exa+0.3*old+0.2*ca
    y=(1/(1+np.exp(-lo))+rng.normal(0,0.04,n)>0.5).astype(int)
    return pd.DataFrame({'age':age,'sex':sex,'cp':cp,'trestbps':tbp,'chol':cho,
        'fbs':fbs,'restecg':rec,'thalach':tha,'exang':exa,
        'oldpeak':old,'slope':slo,'ca':ca,'thal':thl,'target':y})


def gen_parkinsons(n=N_SAMPLES):
    rng=np.random.default_rng(RANDOM_STATE+2)
    fo=rng.normal(154,42,n).clip(80,260); fhi=rng.normal(197,73,n).clip(100,600)
    flo=rng.normal(116,44,n).clip(65,240); jp=rng.exponential(0.006,n).clip(0.001,0.08)
    ja=rng.exponential(3.8e-5,n).clip(1e-6,3e-4); rap=rng.exponential(0.003,n).clip(0,0.05)
    ppq=rng.exponential(0.003,n).clip(0,0.05); ddp=rng.exponential(0.010,n).clip(0,0.16)
    sh=rng.exponential(0.03,n).clip(0.009,0.2); shd=rng.exponential(0.28,n).clip(0.085,2.0)
    a3=rng.exponential(0.016,n).clip(0,0.1); a5=rng.exponential(0.018,n).clip(0,0.14)
    apq=rng.exponential(0.024,n).clip(0,0.14); dda=rng.exponential(0.047,n).clip(0,0.3)
    nhr=rng.exponential(0.025,n).clip(0.0006,0.3); hnr=rng.normal(21.9,4.4,n).clip(8,34)
    rpde=rng.normal(0.50,0.10,n).clip(0.26,0.69); dfa=rng.normal(0.72,0.05,n).clip(0.57,0.83)
    sp1=rng.normal(-5.68,1.09,n).clip(-8,-2); sp2=rng.normal(0.226,0.082,n).clip(0.05,0.45)
    d2=rng.normal(2.38,0.38,n).clip(1.4,3.7); ppe=rng.normal(0.207,0.09,n).clip(0.04,0.53)
    lo=-3.0+60*jp+15*sh-0.03*hnr+5*nhr+2*rpde-0.2*sp1+3*ppe
    y=(1/(1+np.exp(-lo))+rng.normal(0,0.04,n)>0.5).astype(int)
    return pd.DataFrame({'MDVP:Fo(Hz)':fo,'MDVP:Fhi(Hz)':fhi,'MDVP:Flo(Hz)':flo,
        'MDVP:Jitter(%)':jp,'MDVP:Jitter(Abs)':ja,'MDVP:RAP':rap,'MDVP:PPQ':ppq,
        'Jitter:DDP':ddp,'MDVP:Shimmer':sh,'MDVP:Shimmer(dB)':shd,'Shimmer:APQ3':a3,
        'Shimmer:APQ5':a5,'MDVP:APQ':apq,'Shimmer:DDA':dda,'NHR':nhr,'HNR':hnr,
        'RPDE':rpde,'DFA':dfa,'spread1':sp1,'spread2':sp2,'D2':d2,'PPE':ppe,'status':y})


def gen_ckd(n=N_SAMPLES):
    rng=np.random.default_rng(RANDOM_STATE+3)
    age=rng.integers(2,90,n).astype(float); bp=rng.normal(76,14,n).clip(50,180)
    sg=rng.choice([1.005,1.010,1.015,1.020,1.025],n).astype(float)
    al=rng.integers(0,5,n).astype(float); su=rng.integers(0,5,n).astype(float)
    rbc=rng.integers(0,2,n).astype(float); bu=rng.normal(57,50,n).clip(1.5,390)
    sc=rng.normal(3.07,5.1,n).clip(0.4,76); sod=rng.normal(137,10,n).clip(111,163)
    pot=rng.normal(4.6,1.5,n).clip(2.5,47); hem=rng.normal(12.5,2.6,n).clip(3.1,17.8)
    pcv=rng.normal(38,8.7,n).clip(9,54); wbc=rng.normal(8406,3040,n).clip(3800,26400)
    rbc2=rng.normal(4.7,1.1,n).clip(2.1,8.0); htn=(rng.random(n)<0.40).astype(float)
    dm=(rng.random(n)<0.25).astype(float)
    lo=-3.0+0.02*age+0.02*al+0.08*sc+0.01*bu-0.1*hem+0.8*htn+0.6*dm
    y=(1/(1+np.exp(-lo))+rng.normal(0,0.04,n)>0.5).astype(int)
    return pd.DataFrame({'age':age,'bp':bp,'sg':sg,'al':al,'su':su,'rbc':rbc,
        'bu':bu,'sc':sc,'sod':sod,'pot':pot,'hemo':hem,'pcv':pcv,
        'wbcc':wbc,'rbcc':rbc2,'htn':htn,'dm':dm,'classification':y})


def gen_liver(n=N_SAMPLES):
    rng=np.random.default_rng(RANDOM_STATE+4)
    age=rng.integers(4,90,n).astype(float); gen=rng.integers(0,2,n).astype(float)
    tb=rng.exponential(3.3,n).clip(0.4,75); db=rng.exponential(1.5,n).clip(0.1,19.7)
    alk=rng.normal(290,209,n).clip(63,2110); sgt=rng.exponential(80,n).clip(10,2000)
    sgo=rng.exponential(80,n).clip(10,4929); tp=rng.normal(6.5,1.0,n).clip(2.7,9.6)
    alb=rng.normal(3.1,0.7,n).clip(0.9,5.5); agr=rng.normal(1.0,0.3,n).clip(0.3,2.8)
    lo=-1.0+0.01*age+0.2*gen+0.05*tb+0.002*alk+0.001*sgt+0.001*sgo-0.2*alb
    y=(1/(1+np.exp(-lo))+rng.normal(0,0.04,n)>0.5).astype(int)
    return pd.DataFrame({'Age':age,'Gender':gen,'Total_Bilirubin':tb,'Direct_Bilirubin':db,
        'Alkaline_Phosphotase':alk,'Alamine_Aminotransferase':sgt,
        'Aspartate_Aminotransferase':sgo,'Total_Protiens':tp,
        'Albumin':alb,'Albumin_and_Globulin_Ratio':agr,'Dataset':y})


def gen_stroke(n=N_SAMPLES):
    rng=np.random.default_rng(RANDOM_STATE+5)
    age=rng.integers(1,82,n).astype(float); htn=(rng.random(n)<0.30).astype(float)
    hrtd=(rng.random(n)<0.10).astype(float); gluc=rng.normal(106,45,n).clip(55,272)
    bmi=rng.normal(28.9,7.8,n).clip(10,60); smok=rng.integers(0,3,n).astype(float)
    ever_m=(rng.random(n)<0.65).astype(float); work=rng.integers(0,4,n).astype(float)
    resid=rng.integers(0,2,n).astype(float); gender=(rng.random(n)<0.55).astype(float)
    lo=-7.0+0.07*age+1.2*htn+0.8*hrtd+0.01*gluc+0.04*bmi+0.4*smok
    y=(1/(1+np.exp(-lo))+rng.normal(0,0.04,n)>0.5).astype(int)
    return pd.DataFrame({'gender':gender,'age':age,'hypertension':htn,
        'heart_disease':hrtd,'ever_married':ever_m,'work_type':work,
        'Residence_type':resid,'avg_glucose_level':gluc,'bmi':bmi,
        'smoking_status':smok,'stroke':y})


def gen_breast_cancer(n=N_SAMPLES):
    rng=np.random.default_rng(RANDOM_STATE+6)
    mal=rng.random(n)<0.37
    radius=np.where(mal,rng.normal(17.5,3,n),rng.normal(12.1,2,n)).clip(7,28)
    texture=np.where(mal,rng.normal(21.6,4,n),rng.normal(17.9,4,n)).clip(9,40)
    perim=np.where(mal,rng.normal(115,20,n),rng.normal(78,11,n)).clip(44,190)
    area=np.where(mal,rng.normal(978,360,n),rng.normal(462,133,n)).clip(140,2500)
    smooth=np.where(mal,rng.normal(0.102,0.015,n),rng.normal(0.092,0.014,n)).clip(0.05,0.16)
    compact=np.where(mal,rng.normal(0.145,0.05,n),rng.normal(0.080,0.04,n)).clip(0.01,0.35)
    concav=np.where(mal,rng.normal(0.161,0.07,n),rng.normal(0.046,0.05,n)).clip(0,0.43)
    concpts=np.where(mal,rng.normal(0.088,0.03,n),rng.normal(0.025,0.02,n)).clip(0,0.2)
    symm=np.where(mal,rng.normal(0.193,0.03,n),rng.normal(0.174,0.03,n)).clip(0.1,0.3)
    frac=np.where(mal,rng.normal(0.063,0.007,n),rng.normal(0.062,0.007,n)).clip(0.04,0.097)
    y=mal.astype(int)
    return pd.DataFrame({'radius_mean':radius,'texture_mean':texture,'perimeter_mean':perim,
        'area_mean':area,'smoothness_mean':smooth,'compactness_mean':compact,
        'concavity_mean':concav,'concave_points_mean':concpts,'symmetry_mean':symm,
        'fractal_dimension_mean':frac,'diagnosis':y})


def gen_thyroid(n=N_SAMPLES):
    rng=np.random.default_rng(RANDOM_STATE+7)
    age=rng.integers(18,80,n).astype(float); gender=rng.integers(0,2,n).astype(float)
    tsh=rng.exponential(2.5,n).clip(0.005,530); t3=rng.normal(2.0,0.6,n).clip(0.2,5.5)
    tt4=rng.normal(95,30,n).clip(15,300); t4u=rng.normal(0.95,0.14,n).clip(0.5,2.2)
    fti=rng.normal(100,32,n).clip(8,600); on_meds=(rng.random(n)<0.20).astype(float)
    lo=-3.0+0.03*tsh-0.01*tt4-0.5*t3+0.4*gender+0.01*age+0.6*on_meds
    y=(1/(1+np.exp(-lo))+rng.normal(0,0.04,n)>0.5).astype(int)
    return pd.DataFrame({'age':age,'gender':gender,'TSH':tsh,'T3':t3,
        'TT4':tt4,'T4U':t4u,'FTI':fti,'on_thyroxine':on_meds,'target':y})


def gen_pcos(n=N_SAMPLES):
    rng=np.random.default_rng(RANDOM_STATE+8)
    age=rng.integers(15,45,n).astype(float); bmi=rng.normal(26,6,n).clip(13,55)
    cycle=rng.integers(0,2,n).astype(float); amh=rng.exponential(3.5,n).clip(0.2,20)
    fsh=rng.normal(5.5,2,n).clip(0.5,15); lh=rng.normal(6.0,4,n).clip(0.2,25)
    fsh_lh=lh/(fsh+0.01); waist=rng.normal(82,14,n).clip(55,130)
    weight=rng.normal(65,15,n).clip(35,130); hir=(rng.random(n)<0.35).astype(float)
    skin=(rng.random(n)<0.30).astype(float); hair_l=(rng.random(n)<0.25).astype(float)
    follicle=rng.integers(5,32,n).astype(float); blood_g=rng.normal(95,20,n).clip(60,200)
    lo=-2.5+0.04*bmi+0.5*cycle+0.1*amh+0.3*fsh_lh+0.8*hir+0.01*follicle
    y=(1/(1+np.exp(-lo))+rng.normal(0,0.04,n)>0.5).astype(int)
    return pd.DataFrame({'Age':age,'BMI':bmi,'Cycle_Regularity':cycle,'AMH':amh,
        'FSH':fsh,'LH':lh,'FSH_LH_Ratio':fsh_lh,'Waist':waist,'Weight':weight,
        'Hirsutism':hir,'Skin_Darkening':skin,'Hair_Loss':hair_l,
        'Follicle_Count':follicle,'Blood_Glucose':blood_g,'PCOS':y})


def gen_anemia(n=N_SAMPLES):
    rng=np.random.default_rng(RANDOM_STATE+9)
    gender=rng.integers(0,2,n).astype(float); hemog=rng.normal(13.5,2.5,n).clip(4,18)
    mch=rng.normal(27.5,4,n).clip(14,40); mchc=rng.normal(33.5,2,n).clip(25,40)
    mcv=rng.normal(85,12,n).clip(55,115); rbc_c=rng.normal(4.7,0.7,n).clip(2.0,7)
    wbc_c=rng.normal(7000,2000,n).clip(2000,15000); platelet=rng.normal(260,80,n).clip(50,650)
    lo=-2.0-0.4*hemog-0.05*mcv-0.1*mch+0.3*gender
    y=(1/(1+np.exp(-lo))+rng.normal(0,0.04,n)>0.5).astype(int)
    return pd.DataFrame({'Gender':gender,'Hemoglobin':hemog,'MCH':mch,'MCHC':mchc,
        'MCV':mcv,'RBC_Count':rbc_c,'WBC_Count':wbc_c,'Platelets':platelet,'Result':y})


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def train_all_models():
    all_metrics = {}
    print("\n" + "="*70)
    print("  AI MULTI-DISEASE RISK ASSESSMENT — MODEL TRAINING v2.0")
    print("="*70)

    tasks = [
        (gen_diabetes,      ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                              'Insulin','BMI','DiabetesPedigreeFunction','Age'],
                             'Outcome','diabetes','Diabetes'),
        (gen_heart,         ['age','sex','cp','trestbps','chol','fbs','restecg',
                              'thalach','exang','oldpeak','slope','ca','thal'],
                             'target','heart','Heart Disease'),
        (gen_parkinsons,    ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)',
                              'MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ',
                              'Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)',
                              'Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA',
                              'NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE'],
                             'status','parkinsons',"Parkinson's"),
        (gen_ckd,           ['age','bp','sg','al','su','rbc','bu','sc','sod',
                              'pot','hemo','pcv','wbcc','rbcc','htn','dm'],
                             'classification','ckd','Kidney Disease'),
        (gen_liver,         ['Age','Gender','Total_Bilirubin','Direct_Bilirubin',
                              'Alkaline_Phosphotase','Alamine_Aminotransferase',
                              'Aspartate_Aminotransferase','Total_Protiens',
                              'Albumin','Albumin_and_Globulin_Ratio'],
                             'Dataset','liver','Liver Disease'),
        (gen_stroke,        ['gender','age','hypertension','heart_disease',
                              'ever_married','work_type','Residence_type',
                              'avg_glucose_level','bmi','smoking_status'],
                             'stroke','stroke','Stroke Risk'),
        (gen_breast_cancer, ['radius_mean','texture_mean','perimeter_mean','area_mean',
                              'smoothness_mean','compactness_mean','concavity_mean',
                              'concave_points_mean','symmetry_mean','fractal_dimension_mean'],
                             'diagnosis','breast_cancer','Breast Cancer'),
        (gen_thyroid,       ['age','gender','TSH','T3','TT4','T4U','FTI','on_thyroxine'],
                             'target','thyroid','Thyroid Disease'),
        (gen_pcos,          ['Age','BMI','Cycle_Regularity','AMH','FSH','LH',
                              'FSH_LH_Ratio','Waist','Weight','Hirsutism',
                              'Skin_Darkening','Hair_Loss','Follicle_Count','Blood_Glucose'],
                             'PCOS','pcos','PCOS'),
        (gen_anemia,        ['Gender','Hemoglobin','MCH','MCHC','MCV',
                              'RBC_Count','WBC_Count','Platelets'],
                             'Result','anemia','Anemia'),
    ]

    print()
    for gen_fn, feat_cols, tgt_col, key, name in tasks:
        try:
            df = gen_fn()
            metrics = train_save(df, feat_cols, tgt_col, key, name)
            all_metrics[key] = metrics
        except Exception as e:
            print(f"  ❌ FAILED: {name} — {e}")

    with open(os.path.join(MODELS_DIR, 'all_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n  ✅ {len(all_metrics)}/10 models trained and saved successfully!")
    print("="*70 + "\n")
    return all_metrics


if __name__ == "__main__":
    train_all_models()

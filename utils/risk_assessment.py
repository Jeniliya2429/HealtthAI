"""
=============================================================================
  AI Multi-Disease Risk Assessment System
  FILE: utils/risk_assessment.py  |  v2.0
  Handles: Risk classification, Health Score, Diet Recommendations,
           Cross-disease amplification, Risk progression
=============================================================================
"""
import numpy as np

# ─── THRESHOLDS ───────────────────────────────────────────────────────────────
DEFAULT_THRESHOLDS = {"low_upper": 0.33, "med_upper": 0.66}

AGE_THRESHOLDS = {
    "child"  : {"low_upper": 0.25, "med_upper": 0.55},
    "young"  : {"low_upper": 0.35, "med_upper": 0.68},
    "adult"  : {"low_upper": 0.33, "med_upper": 0.66},
    "senior" : {"low_upper": 0.28, "med_upper": 0.60},
    "elderly": {"low_upper": 0.22, "med_upper": 0.52},
}

CROSS_DISEASE_AMPLIFIER = {
    ("diabetes",  "heart")       : {"threshold": 0.60, "boost": 0.06},
    ("diabetes",  "stroke")      : {"threshold": 0.60, "boost": 0.05},
    ("heart",     "stroke")      : {"threshold": 0.65, "boost": 0.05},
    ("heart",     "ckd")         : {"threshold": 0.65, "boost": 0.04},
    ("ckd",       "anemia")      : {"threshold": 0.60, "boost": 0.06},
    ("ckd",       "liver")       : {"threshold": 0.60, "boost": 0.04},
    ("thyroid",   "pcos")        : {"threshold": 0.55, "boost": 0.04},
    ("anemia",    "parkinsons")  : {"threshold": 0.60, "boost": 0.03},
}

DISEASE_WEIGHTS = {
    "heart": 0.18, "stroke": 0.15, "diabetes": 0.14, "breast_cancer": 0.12,
    "ckd": 0.10, "liver": 0.09, "parkinsons": 0.08,
    "thyroid": 0.06, "pcos": 0.05, "anemia": 0.03,
}

# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────

def get_age_group(age: int) -> str:
    if age < 18:   return "child"
    elif age <= 35: return "young"
    elif age <= 55: return "adult"
    elif age <= 70: return "senior"
    else:           return "elderly"


def classify_risk(probability: float, age: int = 40) -> dict:
    """Convert probability [0,1] to risk level with metadata."""
    thresholds = AGE_THRESHOLDS.get(get_age_group(age), DEFAULT_THRESHOLDS)
    prob = float(np.clip(probability, 0.0, 1.0))

    if prob <= thresholds["low_upper"]:
        level, label, emoji, color = "Low", "Low Risk", "🟢", "#2ecc71"
        conf = round((1.0 - prob / thresholds["low_upper"]) * 100, 1)
    elif prob <= thresholds["med_upper"]:
        level, label, emoji, color = "Medium", "Medium Risk", "🟡", "#f39c12"
        mid  = (thresholds["low_upper"] + thresholds["med_upper"]) / 2
        conf = round((1.0 - abs(prob - mid) / (thresholds["med_upper"] - thresholds["low_upper"])) * 100, 1)
    else:
        level, label, emoji, color = "High", "High Risk", "🔴", "#e74c3c"
        conf = round((prob - thresholds["med_upper"]) / (1.0 - thresholds["med_upper"]) * 100, 1)

    return {
        "level": level, "label": label, "emoji": emoji, "color": color,
        "probability": round(prob, 4),
        "probability_pct": round(prob * 100, 2),
        "confidence": round(min(conf, 100.0), 1),
        "age_group": get_age_group(age),
    }


def apply_cross_disease_amplifier(probs: dict) -> dict:
    """Boost probabilities based on comorbidity relationships."""
    adjusted = dict(probs)
    for (src, dst), rule in CROSS_DISEASE_AMPLIFIER.items():
        if src in adjusted and dst in adjusted:
            if adjusted[src] >= rule["threshold"]:
                boost = rule["boost"] * adjusted[src]
                adjusted[dst] = min(1.0, adjusted[dst] + boost)
    return adjusted


def compute_health_score(probs: dict) -> dict:
    """Compute overall health score 0-100 from disease probabilities."""
    try:
        active = {d: DISEASE_WEIGHTS[d] for d in probs if d in DISEASE_WEIGHTS}
        if not active:
            return {"score": 50.0, "grade": "N/A", "color": "#95a5a6"}
        total_w = sum(active.values())
        weighted_risk = sum(probs[d] * w for d, w in active.items()) / total_w
        score = round((1.0 - weighted_risk) * 100, 1)
        if score >= 80: grade, color = "Excellent", "#27ae60"
        elif score >= 65: grade, color = "Good",    "#2ecc71"
        elif score >= 50: grade, color = "Fair",    "#f39c12"
        elif score >= 35: grade, color = "Poor",    "#e67e22"
        else:              grade, color = "Critical","#e74c3c"
        return {"score": score, "grade": grade, "color": color}
    except Exception:
        return {"score": 50.0, "grade": "N/A", "color": "#95a5a6"}


def simulate_risk_progression(current_prob: float, years: int = 5) -> list:
    """Simulate 5-year risk trend if untreated (logistic growth model)."""
    try:
        rng = np.random.default_rng(42)
        base_growth = 0.05 if current_prob < 0.5 else 0.03
        progression, prob = [], current_prob
        for yr in range(years + 1):
            noise = rng.normal(0, 0.005)
            progression.append((yr, round(min(1.0, max(0.0, prob + noise)), 4)))
            prob = min(0.99, prob + base_growth * (1 - prob))
        return progression
    except Exception:
        return [(i, current_prob) for i in range(years + 1)]


# ─── HEALTH TIPS ──────────────────────────────────────────────────────────────

def get_health_tips(disease_key: str, risk_level: str) -> list:
    tips = {
        "diabetes": {
            "Low"   : ["Keep active with 30 min walks daily 🏃", "Maintain a low-sugar balanced diet", "Annual glucose screening is still wise"],
            "Medium": ["Monitor fasting blood sugar weekly", "Cut sugary drinks and white bread from your diet", "Aim for 150 min cardio per week", "Consult a doctor for HbA1c test"],
            "High"  : ["See a doctor or diabetologist promptly", "Track blood glucose daily with a glucometer", "Follow a strict diabetic meal plan", "Medication review may be needed"],
        },
        "heart": {
            "Low"   : ["Great cardiovascular indicators! ❤️", "Keep cholesterol in check", "Stay smoke-free and limit alcohol"],
            "Medium": ["Monitor blood pressure regularly", "Reduce sodium and saturated fats", "150 min aerobic activity per week", "Manage stress through yoga or meditation"],
            "High"  : ["Seek cardiac evaluation soon", "Strictly monitor BP and cholesterol", "Avoid smoking and heavy alcohol immediately", "An ECG or stress test is recommended"],
        },
        "parkinsons": {
            "Low"   : ["Voice biomarkers look healthy 🎵", "Regular aerobic exercise protects brain health", "Stay socially and mentally active"],
            "Medium": ["Consider neurological follow-up", "Practice vocal exercises daily", "Dancing and cycling help motor function"],
            "High"  : ["See a neurologist for proper evaluation", "Structured physiotherapy can help greatly", "Don't self-diagnose — get professional assessment"],
        },
        "ckd": {
            "Low"   : ["Kidneys are healthy! Stay hydrated 💧", "Maintain healthy blood pressure", "Limit NSAID pain relievers"],
            "Medium": ["Monitor creatinine every 3-6 months", "Reduce processed and high-protein meals", "Control blood pressure and sugar"],
            "High"  : ["See a nephrologist urgently", "Follow a strict renal diet", "Monitor fluid intake and electrolytes carefully"],
        },
        "liver": {
            "Low"   : ["Liver function looks great! 🌿", "Limit alcohol to recommended guidelines", "Maintain a healthy body weight"],
            "Medium": ["Reduce alcohol significantly", "Avoid fatty and processed food", "Get a liver function panel (LFT) test"],
            "High"  : ["See a hepatologist as soon as possible", "Strict alcohol avoidance is essential", "Ultrasound or liver biopsy may be needed"],
        },
        "stroke": {
            "Low"   : ["Low stroke risk — keep it up! 🧠", "Maintain blood pressure below 130/80", "Stay physically active"],
            "Medium": ["Control blood pressure strictly", "Quit smoking if you smoke", "Manage atrial fibrillation if diagnosed"],
            "High"  : ["Seek immediate medical consultation", "Control BP, cholesterol, and blood sugar together", "Anticoagulant therapy may be discussed with doctor"],
        },
        "breast_cancer": {
            "Low"   : ["Low indicators — great! 💗", "Monthly self-breast exam recommended", "Annual mammogram from age 40"],
            "Medium": ["Schedule a clinical breast exam soon", "Reduce alcohol and maintain healthy weight", "Discuss genetic screening (BRCA) with doctor"],
            "High"  : ["Book an oncology consultation urgently", "Imaging (mammogram/ultrasound/MRI) is essential", "Early detection is key — don't delay"],
        },
        "thyroid": {
            "Low"   : ["Thyroid function looks balanced ⚖️", "Ensure adequate iodine in diet", "Annual TSH check-up is good practice"],
            "Medium": ["Get TSH, T3, T4 blood panel done", "Watch for fatigue, weight changes, or mood shifts", "Selenium-rich foods support thyroid health"],
            "High"  : ["See an endocrinologist for full thyroid panel", "Medication (levothyroxine/methimazole) may be needed", "Avoid excessive iodine supplements"],
        },
        "pcos": {
            "Low"   : ["Hormonal indicators look balanced 🌸", "Maintain regular exercise routine", "Track menstrual cycle monthly"],
            "Medium": ["Get FSH, LH, and AMH blood tests", "Reduce refined carbs and sugars", "Regular exercise helps regulate hormones"],
            "High"  : ["See a gynecologist or endocrinologist", "Inositol supplements may help (consult doctor)", "Manage weight — even 5% reduction helps PCOS significantly"],
        },
        "anemia": {
            "Low"   : ["Blood health looks good! 🩸", "Ensure adequate iron, B12, and folate", "Eat leafy greens regularly"],
            "Medium": ["Get a complete blood count (CBC) done", "Increase iron-rich foods (lentils, spinach, meat)", "Pair iron foods with Vitamin C for better absorption"],
            "High"  : ["See a doctor for CBC and iron studies urgently", "Iron infusion or B12 injections may be needed", "Identify the underlying cause (bleeding, diet, etc.)"],
        },
    }
    return tips.get(disease_key, {}).get(risk_level, ["Maintain a healthy lifestyle! 🌟"])


# ─── DIET RECOMMENDATIONS ─────────────────────────────────────────────────────

def get_diet_recommendations(disease_key: str, risk_level: str) -> dict:
    """
    Returns comprehensive diet plan with:
    - foods_to_eat (list of food groups + examples)
    - foods_to_avoid (list)
    - meal_plan (breakfast, lunch, dinner, snack)
    - nutrition_targets (key nutrients to watch)
    - supplements (suggested — consult doctor first)
    - hydration tip
    """
    diets = {
        # ── DIABETES ──────────────────────────────────────────────────────────
        "diabetes": {
            "Low": {
                "headline"   : "Maintain Your Blood Sugar Balance 🥗",
                "foods_eat"  : ["🥬 Leafy greens — spinach, kale, methi",
                                "🫐 Low-GI fruits — berries, apple, pear",
                                "🌾 Whole grains — oats, brown rice, quinoa",
                                "🥚 Lean protein — eggs, lentils, tofu, chicken",
                                "🥑 Healthy fats — avocado, walnuts, olive oil"],
                "foods_avoid": ["❌ White rice and white bread (high GI)",
                                "❌ Sugary drinks — cola, juice, energy drinks",
                                "❌ Sweets, cakes, pastries",
                                "❌ Deep-fried snacks"],
                "meal_plan"  : {
                    "Breakfast": "Oats with chia seeds + handful of berries + green tea",
                    "Lunch"    : "Brown rice + dal + sautéed vegetables + curd",
                    "Dinner"   : "Grilled chicken/paneer + quinoa + salad",
                    "Snack"    : "Apple slices with almond butter or roasted chickpeas",
                },
                "nutrition"  : ["Keep carbs < 45% of total calories", "Fibre > 25g/day", "Sugar < 25g/day"],
                "hydration"  : "Drink 8–10 glasses of water daily. Avoid sweetened beverages.",
                "supplements": ["Chromium (ask doctor)", "Magnesium", "Vitamin D"],
            },
            "Medium": {
                "headline"   : "Pre-Diabetic Control Diet 🩺",
                "foods_eat"  : ["🥦 Non-starchy vegetables — broccoli, cauliflower, cucumber",
                                "🐟 Fatty fish — salmon, sardines (omega-3)",
                                "🫘 Legumes — lentils, chickpeas, kidney beans",
                                "🌰 Nuts — almonds, walnuts (small portions)",
                                "🥛 Low-fat dairy — curd, skimmed milk"],
                "foods_avoid": ["❌ Refined carbohydrates entirely",
                                "❌ Packaged cereals and granola bars",
                                "❌ Fruit juices (even natural — eat whole fruit)",
                                "❌ High-fat red meat"],
                "meal_plan"  : {
                    "Breakfast": "Vegetable omelette (2 eggs) + 1 slice multigrain toast + herbal tea",
                    "Lunch"    : "Lentil soup + mixed salad + 1 small whole wheat roti",
                    "Dinner"   : "Grilled fish + steamed broccoli + 1/2 cup brown rice",
                    "Snack"    : "Handful of mixed nuts + cucumber slices",
                },
                "nutrition"  : ["Carbs < 35% of calories", "Fibre > 30g/day", "Zero added sugar"],
                "hydration"  : "10–12 glasses of water. Try cinnamon or fenugreek-infused water.",
                "supplements": ["Berberine (doctor-advised)", "Alpha-lipoic acid", "Vitamin D3"],
            },
            "High": {
                "headline"   : "Strict Diabetic Management Diet 🚨",
                "foods_eat"  : ["🥬 Only low-GI vegetables — all leafy greens",
                                "🫘 Lentils and legumes as protein source",
                                "🐓 Lean protein only — chicken breast, fish, tofu",
                                "🫒 Olive oil and coconut oil in small quantities"],
                "foods_avoid": ["❌ All sugars and sweeteners (including honey, jaggery)",
                                "❌ All refined grains — maida, white rice",
                                "❌ Fruit juices and most fruits except berries",
                                "❌ Alcohol completely", "❌ Fried foods entirely"],
                "meal_plan"  : {
                    "Breakfast": "Moong dal chilla (2 pieces) + mint chutney + green tea (no sugar)",
                    "Lunch"    : "Small portion brown rice + sambar + stir-fried low-GI veggies",
                    "Dinner"   : "Grilled fish/tofu + salad + vegetable soup (no cream)",
                    "Snack"    : "Roasted seeds (pumpkin/sunflower) + herbal tea",
                },
                "nutrition"  : ["Carbs < 25% of calories", "Protein 30%", "Fibre > 35g/day", "Zero sugar"],
                "hydration"  : "12+ glasses of water. Methi water in morning helps glucose control.",
                "supplements": ["Chromium picolinate (doctor)", "Magnesium", "Vitamin D3+K2"],
            },
        },

        # ── HEART DISEASE ─────────────────────────────────────────────────────
        "heart": {
            "Low": {
                "headline"   : "Heart-Healthy Maintenance Diet ❤️",
                "foods_eat"  : ["🐟 Fatty fish — salmon, tuna (2x/week)", "🫐 Berries and antioxidant-rich fruits", "🥑 Avocado and olive oil", "🌰 Walnuts and flaxseeds", "🥦 Cruciferous vegetables"],
                "foods_avoid": ["❌ Trans fats (processed snacks, cookies)", "❌ Excessive salt (>2g/day)", "❌ Fatty red meats"],
                "meal_plan"  : {"Breakfast": "Oatmeal + flaxseeds + blueberries", "Lunch": "Grilled salmon + salad + olive oil dressing", "Dinner": "Lentil soup + whole grain bread + steamed greens", "Snack": "Walnuts + apple"},
                "nutrition"  : ["Sodium < 2g/day", "Saturated fat < 7%", "Omega-3: 1g/day"],
                "hydration"  : "8 glasses of water. Green tea has cardioprotective benefits.",
                "supplements": ["Omega-3 fish oil", "CoQ10 (ask doctor)", "Magnesium"],
            },
            "Medium": {
                "headline"   : "Cholesterol & BP Control Diet 🩺",
                "foods_eat"  : ["🥬 Dark leafy greens daily", "🍅 Tomatoes, beets (nitrates help BP)", "🫘 Legumes — reduce LDL cholesterol", "🐟 Oily fish 3x/week", "🌾 Oats and barley (beta-glucan)"],
                "foods_avoid": ["❌ Butter, ghee in excess", "❌ Salty snacks, pickles, papad", "❌ Egg yolks (limit to 3/week)", "❌ Coconut milk in excess"],
                "meal_plan"  : {"Breakfast": "Oat porridge + banana + green tea", "Lunch": "Dal + 1 roti + sabzi + buttermilk (no salt)", "Dinner": "Fish curry (light) + brown rice + salad", "Snack": "Mixed seeds + herbal tea"},
                "nutrition"  : ["Sodium < 1.5g/day", "Saturated fat < 5%", "Omega-3 > 2g/day", "Fibre > 30g"],
                "hydration"  : "10 glasses of water. Hibiscus tea helps lower BP naturally.",
                "supplements": ["Plant sterols", "Berberine", "Omega-3", "Vitamin K2"],
            },
            "High": {
                "headline"   : "Cardiac Recovery Diet 🚨",
                "foods_eat"  : ["🥬 Mostly plant-based diet", "🫘 Legumes as primary protein", "🐟 Only fatty fish — no red meat", "🍎 Low-sugar fruits — apples, pears, berries"],
                "foods_avoid": ["❌ All red and processed meat", "❌ Dairy fat — full-fat cheese, butter", "❌ All fried food completely", "❌ Alcohol completely", "❌ Excess caffeine"],
                "meal_plan"  : {"Breakfast": "Oatmeal + chia + berries + herbal tea", "Lunch": "Chickpea salad + whole grain wrap + vegetable soup", "Dinner": "Grilled salmon + steamed broccoli + quinoa", "Snack": "Cucumber + hummus"},
                "nutrition"  : ["Sodium < 1g/day", "No saturated fat", "Omega-3 > 3g/day"],
                "hydration"  : "12 glasses of water minimum. No energy drinks or caffeinated sodas.",
                "supplements": ["Omega-3 (high dose — doctor)", "CoQ10", "Magnesium glycinate", "Vitamin D3"],
            },
        },

        # ── PARKINSON'S ───────────────────────────────────────────────────────
        "parkinsons": {
            "Low"   : {"headline": "Neuroprotective Diet 🧠", "foods_eat": ["🫐 Berries — antioxidants protect neurons", "☕ Coffee (1-2 cups/day — research-backed)", "🥦 Cruciferous vegetables", "🐟 Fatty fish — omega-3 for brain health", "🌰 Nuts and seeds"], "foods_avoid": ["❌ Processed and ultra-processed foods", "❌ Excess saturated fats", "❌ High-protein meals if on Levodopa"], "meal_plan": {"Breakfast": "Blueberry smoothie + oats + coffee", "Lunch": "Grilled fish + salad + olive oil", "Dinner": "Lentil soup + steamed vegetables", "Snack": "Walnuts + green tea"}, "nutrition": ["Antioxidants high", "Omega-3 > 1g/day", "Vitamin E rich foods"], "hydration": "8-10 glasses of water. Constipation is common in Parkinson's — stay hydrated.", "supplements": ["Coenzyme Q10", "Vitamin D", "Omega-3"]},
            "Medium": {"headline": "Anti-Inflammatory Neuro Diet 🩺", "foods_eat": ["🫐 Daily berries — blueberry, strawberry", "🥬 Spinach, kale, broccoli daily", "🐟 Salmon or sardines 3x/week", "☕ Green tea or coffee", "🌿 Turmeric in cooking (curcumin)"], "foods_avoid": ["❌ Dairy in large amounts (may affect Levodopa)", "❌ High-fat red meat", "❌ Alcohol", "❌ Excess iron supplements (neurotoxic)"], "meal_plan": {"Breakfast": "Berry and spinach smoothie + chia seeds + herbal tea", "Lunch": "Brown rice + dal + turmeric vegetable curry", "Dinner": "Grilled salmon + steamed greens + quinoa", "Snack": "Walnuts + blueberries"}, "nutrition": ["Anti-inflammatory focus", "Curcumin 1g/day", "Omega-3 > 2g/day"], "hydration": "10-12 glasses — prevents constipation and maintains medication absorption.", "supplements": ["CoQ10 1200mg (doctor)", "Vitamin D3", "Mucuna pruriens (consult doctor)"]},
            "High"  : {"headline": "Therapeutic Parkinson's Diet 🚨", "foods_eat": ["🫐 Berries at every meal", "🥬 Only plant-based — Mediterranean style", "🐟 Fatty fish daily", "🌿 Anti-inflammatory spices — turmeric, ginger", "🫘 Legumes as protein (not competing with Levodopa)"], "foods_avoid": ["❌ High-protein meals around medication time", "❌ All processed foods", "❌ Alcohol", "❌ Excess dairy"], "meal_plan": {"Breakfast": "Oatmeal + flaxseed + berries (1hr after morning medication)", "Lunch": "Vegetable soup + whole grain bread (low protein)", "Dinner": "Grilled fish + Mediterranean salad (high protein OK at dinner)", "Snack": "Nuts + blueberries"}, "nutrition": ["Protein timing around Levodopa", "Antioxidants at every meal", "Fibre > 35g/day"], "hydration": "12+ glasses. Warm prune juice helps constipation.", "supplements": ["CoQ10 1200mg", "Vitamin D3", "Melatonin for sleep (doctor)"]},
        },

        # ── CKD ───────────────────────────────────────────────────────────────
        "ckd": {
            "Low"   : {"headline": "Kidney-Protective Diet 🫘", "foods_eat": ["💧 Stay well hydrated", "🍎 Apples, blueberries, grapes (low potassium)", "🥗 Cauliflower, cabbage, turnips", "🍳 Egg whites (lower phosphorus than yolk)", "🌾 White rice (lower potassium than brown)"], "foods_avoid": ["❌ Excessive protein (>0.8g/kg/day)", "❌ NSAID pain relievers", "❌ Dark colas (high phosphorus)", "❌ Excessive sodium"], "meal_plan": {"Breakfast": "Egg white omelette + white toast + apple", "Lunch": "White rice + cauliflower curry + cucumber salad", "Dinner": "Grilled chicken (small) + steamed cabbage + white pasta", "Snack": "Apple slices + herbal tea"}, "nutrition": ["Protein 0.8g/kg/day", "Sodium < 2g/day", "Phosphorus moderate"], "hydration": "Follow doctor guidance on fluid intake. Generally 6-8 glasses.", "supplements": ["Vitamin D (kidney patients often deficient — doctor)", "B-complex (water soluble)"]},
            "Medium": {"headline": "Renal Management Diet 🩺", "foods_eat": ["🍎 Low-potassium fruits only — apples, grapes, pineapple", "🥦 Leached vegetables (soak in water to reduce K+)", "🍳 Egg whites", "🍞 White bread (lower phosphorus)", "🫒 Olive oil"], "foods_avoid": ["❌ Bananas, oranges, potatoes (high potassium)", "❌ Nuts and seeds (high phosphorus)", "❌ Dairy in excess (high phosphorus)", "❌ Processed meats (high sodium + phosphorus)"], "meal_plan": {"Breakfast": "Cream of wheat + apple + egg whites + herbal tea", "Lunch": "Stir-fried cauliflower + white rice + lemon herb chicken (small)", "Dinner": "Vegetable soup (low K) + white pasta + cabbage salad", "Snack": "Grapes + herbal tea"}, "nutrition": ["Protein 0.6-0.8g/kg/day", "Potassium < 2g/day", "Phosphorus < 1g/day", "Sodium < 1.5g/day"], "hydration": "Doctor-guided fluid restriction. Track fluid intake carefully.", "supplements": ["Renal vitamin B-complex ONLY", "Vitamin D (doctor prescribed)"]},
            "High"  : {"headline": "Strict Renal Diet 🚨", "foods_eat": ["🥬 Only lowest-potassium vegetables (leached)", "🍎 Small portions of apple/grapes only", "🍳 Egg whites as main protein", "🍞 White refined carbs (lower K and P)", "🫒 Olive/canola oil for calories"], "foods_avoid": ["❌ ALL potassium-rich foods — banana, potato, tomato, avocado", "❌ ALL phosphorus-rich foods — dairy, nuts, beans, colas", "❌ Processed/canned foods (sodium)", "❌ Potassium salt substitutes (dangerous)", "❌ Herbal supplements without nephrologist approval"], "meal_plan": {"Breakfast": "White toast + jam + egg white + apple (peeled) + water", "Lunch": "Boiled white rice + boiled leached cauliflower + 60g chicken", "Dinner": "White pasta + olive oil + steamed leached green beans", "Snack": "Small apple + herbal tea"}, "nutrition": ["Protein 0.5-0.6g/kg/day", "Potassium < 1.5g/day", "Phosphorus < 800mg/day", "Fluid — strictly doctor advised"], "hydration": "STRICTLY follow nephrologist fluid restriction. Track every ml.", "supplements": ["Only nephrologist-prescribed supplements"]},
        },

        # ── LIVER ─────────────────────────────────────────────────────────────
        "liver": {
            "Low"   : {"headline": "Liver-Friendly Diet 🌿", "foods_eat": ["☕ Coffee (2 cups/day — shown to protect liver)", "🫐 Berries — antioxidants reduce liver fat", "🥬 Leafy greens — spinach, kale", "🧄 Garlic (activates liver enzymes)", "🥑 Avocado (glutathione)"], "foods_avoid": ["❌ Alcohol — even small amounts strain liver", "❌ Fatty and fried foods", "❌ Excess fructose (sugary drinks)", "❌ Unnecessary medications/supplements"], "meal_plan": {"Breakfast": "Oatmeal + berries + black coffee", "Lunch": "Grilled fish + spinach salad + olive oil dressing", "Dinner": "Lentil soup + garlic vegetables + brown rice", "Snack": "Handful of walnuts + green tea"}, "nutrition": ["Antioxidants high", "Healthy fats only", "Avoid alcohol completely"], "hydration": "10 glasses water. Dandelion root tea supports liver function.", "supplements": ["Milk thistle (silymarin)", "Vitamin E", "N-acetyl cysteine (NAC)"]},
            "Medium": {"headline": "Liver Repair Diet 🩺", "foods_eat": ["🥬 Cruciferous vegetables — broccoli, Brussels sprouts", "🐟 Fatty fish for omega-3", "🫘 Legumes as protein", "☕ Black coffee 2-3 cups", "🌿 Turmeric — curcumin is hepatoprotective"], "foods_avoid": ["❌ Alcohol completely", "❌ Red and processed meat", "❌ High-fructose corn syrup", "❌ White bread and refined grains"], "meal_plan": {"Breakfast": "Turmeric oatmeal + banana + black coffee", "Lunch": "Salmon + broccoli + quinoa salad", "Dinner": "Lentil dal + stir-fried vegetables + 1 roti", "Snack": "Walnuts + green tea"}, "nutrition": ["No alcohol", "Omega-3 > 2g/day", "Antioxidants at every meal"], "hydration": "12 glasses of water. Beetroot juice is liver-supportive.", "supplements": ["Milk thistle 300mg", "NAC 600mg", "Vitamin E (doctor)", "Omega-3"]},
            "High"  : {"headline": "Liver Disease Therapeutic Diet 🚨", "foods_eat": ["🥬 Mostly plant-based soft foods", "🍌 Easily digestible fruits", "🥣 Oatmeal, cream of wheat (easy to digest)", "🥚 Egg whites (if no encephalopathy)", "🫘 Legumes in moderate amounts"], "foods_avoid": ["❌ Alcohol — ZERO tolerance", "❌ ALL red meat", "❌ Raw seafood (infection risk)", "❌ Salt (prevents fluid retention/ascites)", "❌ Vitamin A supplements (toxic to damaged liver)"], "meal_plan": {"Breakfast": "Plain oatmeal + banana + herbal tea (no caffeine)", "Lunch": "Steamed soft vegetables + boiled dal + white rice", "Dinner": "Vegetable soup + soft-boiled egg whites + toast", "Snack": "Banana or boiled sweet potato"}, "nutrition": ["Sodium < 1g/day", "Small frequent meals (5-6/day)", "Easy-digest protein 1.2g/kg if no encephalopathy"], "hydration": "Doctor-guided. Track fluid if ascites present.", "supplements": ["ONLY hepatologist-approved supplements", "Zinc (often deficient in liver disease)"]},
        },

        # ── STROKE ────────────────────────────────────────────────────────────
        "stroke": {
            "Low"   : {"headline": "Brain & Vascular Protective Diet 🧠", "foods_eat": ["🫐 Berries — anthocyanins protect blood vessels", "🥬 Leafy greens — folate reduces stroke risk", "🐟 Fatty fish — omega-3 reduces clotting risk", "🥑 Avocado — potassium for BP control", "🍅 Tomatoes — lycopene is neuroprotective"], "foods_avoid": ["❌ Excessive sodium — raises BP", "❌ Trans fats in packaged foods", "❌ Heavy alcohol"], "meal_plan": {"Breakfast": "Spinach and berry smoothie + whole grain toast + green tea", "Lunch": "Grilled salmon + mixed salad + avocado", "Dinner": "Mediterranean bowl — chickpeas + vegetables + olive oil", "Snack": "Blueberries + walnuts"}, "nutrition": ["Sodium < 2g/day", "Potassium > 3.5g/day", "Omega-3 > 1g/day"], "hydration": "8-10 glasses of water. Dehydration increases blood clot risk.", "supplements": ["Omega-3", "Vitamin D", "Folate (B9)", "Magnesium"]},
            "Medium": {"headline": "Stroke Prevention Diet 🩺", "foods_eat": ["🥬 Daily leafy greens", "🍅 Tomatoes, red peppers (lycopene)", "🫘 Legumes for fibre and potassium", "🐟 Fatty fish 3-4x/week", "🌰 Walnuts and flaxseeds"], "foods_avoid": ["❌ Processed meats — high sodium", "❌ Cheese in excess", "❌ Alcohol", "❌ Saturated fat from red meat"], "meal_plan": {"Breakfast": "Oats + ground flaxseed + berries + herbal tea", "Lunch": "Lentil soup + whole grain bread + tomato salad", "Dinner": "Grilled tuna + steamed broccoli + quinoa", "Snack": "Walnuts + unsalted sunflower seeds"}, "nutrition": ["Sodium < 1.5g/day", "DASH diet pattern", "Omega-3 > 2g/day"], "hydration": "10-12 glasses of water daily.", "supplements": ["Omega-3 (high-dose)", "Magnesium", "Vitamin B complex", "Vitamin D3"]},
            "High"  : {"headline": "High-Priority Stroke Prevention Diet 🚨", "foods_eat": ["🥬 Plant-dominant diet — Mediterranean style", "🐟 Fatty fish as primary protein", "🫘 Legumes daily", "🍎 Low-sugar fruits", "🌿 Anti-inflammatory herbs — ginger, turmeric"], "foods_avoid": ["❌ Salt completely (use herbs/lemon instead)", "❌ All processed foods", "❌ Red meat", "❌ Alcohol completely", "❌ Sugary foods (glucose spikes cause vessel damage)"], "meal_plan": {"Breakfast": "Berry spinach smoothie + chia seeds + herbal tea", "Lunch": "Mediterranean salad + grilled salmon + whole grain bread", "Dinner": "Chickpea curry + brown rice + cucumber raita (no salt)", "Snack": "Celery sticks + hummus"}, "nutrition": ["Sodium < 1g/day", "DASH + Mediterranean hybrid", "Blood pressure diet focus"], "hydration": "12+ glasses water. Hibiscus and green tea are vascular-protective.", "supplements": ["Omega-3 (3g — doctor)", "CoQ10", "Vitamin D3+K2", "Magnesium glycinate"]},
        },

        # ── BREAST CANCER ─────────────────────────────────────────────────────
        "breast_cancer": {
            "Low"   : {"headline": "Cancer-Preventive Diet 💗", "foods_eat": ["🥦 Cruciferous vegetables — broccoli, cauliflower (sulforaphane)", "🫐 Berries — antioxidants", "🌱 Flaxseeds (phytoestrogens — balance hormones)", "🧄 Garlic and onions", "🍵 Green tea — EGCG is anticancer"], "foods_avoid": ["❌ Alcohol (even moderate increases risk)", "❌ Processed meats", "❌ High-fat dairy"], "meal_plan": {"Breakfast": "Flaxseed oatmeal + berries + green tea", "Lunch": "Cruciferous vegetable stir-fry + tofu + brown rice", "Dinner": "Grilled salmon + broccoli + quinoa", "Snack": "Walnuts + green tea"}, "nutrition": ["Phytoestrogens from flax", "Antioxidants daily", "Fibre > 30g/day"], "hydration": "10 glasses water + 2 cups green tea.", "supplements": ["Vitamin D3 (low levels linked to risk)", "Omega-3", "Folate"]},
            "Medium": {"headline": "Hormone-Balancing Anti-Cancer Diet 🩺", "foods_eat": ["🥦 Broccoli sprouts daily (highest sulforaphane)", "🌱 Flaxseeds + sesame (lignans)", "🍵 Green tea 3+ cups/day", "🧄 Garlic daily", "🫐 Dark berries daily"], "foods_avoid": ["❌ Alcohol completely", "❌ Red and processed meat", "❌ High-fat dairy — butter, cheese", "❌ Refined sugars"], "meal_plan": {"Breakfast": "Broccoli sprout smoothie + flaxseed + berries", "Lunch": "Tofu + cruciferous veggie bowl + sesame dressing", "Dinner": "Lentil soup + whole grains + garlic vegetables", "Snack": "Green tea + walnuts"}, "nutrition": ["Lignans from flax daily", "No alcohol", "Omega-3 > 2g/day", "Fibre > 35g"], "hydration": "12 glasses water + green tea.", "supplements": ["Vitamin D3 2000–4000 IU (doctor)", "DIM (diindolylmethane — doctor)", "Omega-3", "Melatonin (doctor)"]},
            "High"  : {"headline": "Oncology Support Diet 🚨", "foods_eat": ["🥦 Cruciferous at every meal", "🌱 Flaxseeds in all meals", "🍵 Green tea 4-5 cups/day", "🫘 Plant protein — legumes, tofu, tempeh", "🫐 Berries as daily snack"], "foods_avoid": ["❌ Alcohol — zero tolerance", "❌ All red and processed meat completely", "❌ Full-fat dairy", "❌ Refined carbs and sugar", "❌ Soy supplements (phytoestrogen excess — consult oncologist)"], "meal_plan": {"Breakfast": "Ground flaxseed oatmeal + broccoli sprouts + berries + green tea", "Lunch": "Tofu stir-fry + cruciferous vegetables + brown rice", "Dinner": "Lentil dal + garlic spinach + 1 multigrain roti", "Snack": "Walnuts + green tea + small bowl berries"}, "nutrition": ["Plant-based as primary", "Lignans daily", "Antioxidants every meal", "Fibre > 40g"], "hydration": "12+ glasses water + green tea. Avoid plastic water bottles.", "supplements": ["CONSULT ONCOLOGIST before all supplements", "Vitamin D3 (blood-test guided)", "Omega-3", "Probiotics for gut immune support"]},
        },

        # ── THYROID ───────────────────────────────────────────────────────────
        "thyroid": {
            "Low"   : {"headline": "Thyroid-Supportive Diet ⚖️", "foods_eat": ["🐟 Seafood — iodine-rich (fish, shrimp)", "🥚 Eggs — selenium and iodine", "🌾 Whole grains", "🫘 Legumes", "🥜 Brazil nuts (1-2/day — highest selenium)"], "foods_avoid": ["❌ Raw cruciferous in excess (goitrogens — cook them)", "❌ Soy in excess (affects T4 absorption)", "❌ Gluten (many thyroid patients are sensitive)"], "meal_plan": {"Breakfast": "Scrambled eggs + whole grain toast + green tea", "Lunch": "Grilled fish + cooked broccoli + brown rice", "Dinner": "Lentil soup + cooked vegetables + quinoa", "Snack": "1-2 Brazil nuts + apple"}, "nutrition": ["Iodine from natural food sources", "Selenium 55-200mcg/day", "Zinc rich foods"], "hydration": "8-10 glasses water. Avoid fluoride-heavy water (affects thyroid).", "supplements": ["Selenium 200mcg", "Zinc", "Vitamin D3", "Magnesium"]},
            "Medium": {"headline": "Thyroid Regulation Diet 🩺", "foods_eat": ["🐟 Seafood daily for iodine", "🥚 Eggs (selenium)", "🥜 Brazil nuts (2/day)", "🍗 Lean meat", "🌿 Ashwagandha tea (adaptogen for thyroid)"], "foods_avoid": ["❌ Soy milk, tofu in excess (if hypothyroid)", "❌ Gluten (trial elimination if Hashimoto's suspected)", "❌ Raw kale/cabbage in large amounts", "❌ Processed foods (inflammatory)"], "meal_plan": {"Breakfast": "2 eggs + smoked salmon + gluten-free toast", "Lunch": "Grilled shrimp + cooked vegetables + quinoa", "Dinner": "Chicken soup + cooked spinach + white rice", "Snack": "2 Brazil nuts + pumpkin seeds"}, "nutrition": ["Iodine 150–220mcg/day", "Selenium 200mcg/day", "No gluten trial for 4 weeks"], "hydration": "10-12 glasses filtered water.", "supplements": ["Selenium 200mcg", "Ashwagandha (doctor)", "Vitamin D3", "B-complex", "Zinc 30mg"]},
            "High"  : {"headline": "Thyroid Therapy Diet 🚨", "foods_eat": ["🐟 Iodine-rich seafood daily", "🥚 Eggs daily", "🥜 Brazil nuts (2/day exactly)", "🍗 Lean clean protein", "🌾 Gluten-free grains — rice, quinoa, oats"], "foods_avoid": ["❌ Gluten completely (if Hashimoto's — autoimmune cause)", "❌ Soy entirely if on thyroxine medication", "❌ Raw cruciferous vegetables", "❌ Calcium within 4 hours of thyroid medication", "❌ Coffee within 1 hour of medication (blocks absorption)"], "meal_plan": {"Breakfast": "Gluten-free oats + 2 eggs + green tea (2hrs after medication)", "Lunch": "Grilled fish + roasted vegetables + quinoa", "Dinner": "Chicken + cooked spinach + white rice", "Snack": "2 Brazil nuts + pumpkin seeds + apple"}, "nutrition": ["Gluten-free strictly", "Iodine 150mcg/day (not excess)", "Selenium 200mcg/day", "Take medication timing seriously"], "hydration": "12 glasses filtered water (avoid tap if fluoridated).", "supplements": ["CONSULT ENDOCRINOLOGIST for all supplements", "Selenium 200mcg", "Vitamin D3", "Magnesium glycinate"]},
        },

        # ── PCOS ──────────────────────────────────────────────────────────────
        "pcos": {
            "Low"   : {"headline": "Hormone-Balancing Diet 🌸", "foods_eat": ["🌾 Low-GI whole grains — oats, quinoa", "🥬 Leafy greens daily", "🐟 Fatty fish for anti-inflammation", "🫘 Legumes for protein + fibre", "🌰 Flaxseeds (balances oestrogen)"], "foods_avoid": ["❌ Refined sugar and white flour", "❌ Dairy in excess (IGF-1 may worsen PCOS)", "❌ Alcohol"], "meal_plan": {"Breakfast": "Oats + flaxseed + berries + green tea", "Lunch": "Grilled fish + salad + quinoa", "Dinner": "Dal + sabzi + 1 roti + curd (low-fat)", "Snack": "Handful of nuts + green tea"}, "nutrition": ["Low GI diet", "Anti-inflammatory focus", "Fibre > 30g/day"], "hydration": "10 glasses water + spearmint tea (reduces androgens).", "supplements": ["Inositol (myo-inositol 4g/day)", "Vitamin D3", "Magnesium", "Zinc"]},
            "Medium": {"headline": "PCOS Management Diet 🩺", "foods_eat": ["🥬 Non-starchy vegetables fill half the plate", "🐟 Fatty fish 3x/week", "🫘 Legumes as protein source", "🌰 Flaxseeds + pumpkin seeds", "🍋 Anti-inflammatory herbs — turmeric, cinnamon"], "foods_avoid": ["❌ Dairy — cow's milk, full-fat cheese (try A2 or goat)", "❌ All refined sugars and grains", "❌ Processed packaged foods", "❌ Soy in excess (phytoestrogens)"], "meal_plan": {"Breakfast": "Moong dal chilla + mint chutney + spearmint tea", "Lunch": "Brown rice (small) + dal + sabzi (lot of vegetables)", "Dinner": "Grilled fish + roasted vegetables (no starch)", "Snack": "Flaxseed water + mixed seeds"}, "nutrition": ["Low GI: GI < 50 for all meals", "Dairy-free trial for 4 weeks", "Inositol 4g/day", "Anti-inflammatory"], "hydration": "12 glasses water + spearmint tea 2 cups/day.", "supplements": ["Myo-inositol 4g + D-chiro inositol 1g", "Vitamin D3", "Magnesium 300mg", "NAC 1800mg (doctor)", "Omega-3"]},
            "High"  : {"headline": "Intensive PCOS Reversal Diet 🚨", "foods_eat": ["🥬 Mostly vegetables (low carb approach)", "🐟 Fatty fish daily", "🥚 Eggs (hormone-supporting protein)", "🌰 Seeds — flaxseed, pumpkin, sunflower", "🍋 Anti-inflammatory focus — turmeric, ginger"], "foods_avoid": ["❌ ALL dairy completely (strong PCOS aggravator)", "❌ All refined carbs and sugar", "❌ Soy products", "❌ Alcohol", "❌ Processed and packaged foods entirely"], "meal_plan": {"Breakfast": "Egg veggie omelette + flaxseeds + spearmint tea", "Lunch": "Grilled salmon + large salad + olive oil dressing", "Dinner": "Chicken/tofu + roasted low-GI vegetables", "Snack": "Pumpkin + sunflower seeds + herbal tea"}, "nutrition": ["Very low GI (<40)", "No dairy completely", "Inositol 6g/day", "Protein 30% of calories"], "hydration": "12 glasses water + 2 cups spearmint tea + 1 cup cinnamon water.", "supplements": ["Myo-inositol 4-6g (divided doses)", "Vitamin D3 4000 IU", "Magnesium glycinate 400mg", "NAC 1800mg", "Berberine 1500mg (doctor)", "Omega-3 3g"]},
        },

        # ── ANEMIA ────────────────────────────────────────────────────────────
        "anemia": {
            "Low"   : {"headline": "Blood-Building Maintenance Diet 🩸", "foods_eat": ["🥩 Lean red meat (2-3x/week for heme iron)", "🥬 Spinach, methi (non-heme iron)", "🫘 Lentils and beans", "🥜 Pumpkin seeds (iron-rich)", "🍊 Vitamin C with every iron-rich meal"], "foods_avoid": ["❌ Tea/coffee with meals (blocks iron absorption)", "❌ Calcium with iron-rich meals (competes)", "❌ Refined grains (low in iron)"], "meal_plan": {"Breakfast": "Iron-fortified oats + orange juice + dates", "Lunch": "Dal + spinach sabzi + roti + lemon", "Dinner": "Small portion lean meat + salad + steamed greens", "Snack": "Pumpkin seeds + orange slices"}, "nutrition": ["Iron 18mg/day (women), 8mg (men)", "Vitamin C at every meal", "B12 if vegetarian"], "hydration": "8-10 glasses water. Avoid tea/coffee 1hr before and after meals.", "supplements": ["Vitamin C 500mg with meals", "B12 (especially vegetarians)", "Folate (B9)"]},
            "Medium": {"headline": "Iron-Recovery Diet 🩺", "foods_eat": ["🥩 Red meat 4-5x/week (heme iron — most absorbable)", "🥬 Spinach, beetroot daily", "🫘 Lentils + legumes at every meal", "🍳 Blackstrap molasses", "🍊 Citrus with every meal (vitamin C doubles absorption)"], "foods_avoid": ["❌ Tea, coffee, and cocoa with meals", "❌ Dairy with iron-rich meals", "❌ Antacids (reduce iron absorption)", "❌ Whole grains in excess (phytates block iron)"], "meal_plan": {"Breakfast": "Beetroot + spinach smoothie + dates + orange juice", "Lunch": "Mutton/lentil curry + vitamin C salad + roti", "Dinner": "Liver (if non-veg) / rajma + iron-rich vegetables", "Snack": "Pumpkin seeds + amla (Indian gooseberry)"}, "nutrition": ["Iron 25-30mg/day", "Vitamin C 500mg per meal", "B12 if veg (crucial)", "Folate daily"], "hydration": "10-12 glasses water. Nettle tea is iron-rich.", "supplements": ["Iron bisglycinate (gentle form)", "Vitamin C 1000mg/day", "Vitamin B12 1000mcg", "Folate 400mcg", "Vitamin D3"]},
            "High"  : {"headline": "Severe Anemia Recovery Diet 🚨", "foods_eat": ["🥩 Liver and organ meat (highest iron — non-veg)", "🥬 Spinach, beetroot at every meal", "🫘 Fortified legumes + iron-rich grains", "🍊 Vitamin C source with EVERY single meal", "🌿 Blackstrap molasses (iron + B vitamins)"], "foods_avoid": ["❌ Tea and coffee completely (strong iron blockers)", "❌ Dairy in main meals (take separately)", "❌ Whole grains dominating meals", "❌ Alcohol (damages red blood cells)"], "meal_plan": {"Breakfast": "Spinach + beetroot + orange smoothie + iron-fortified cereal", "Lunch": "Chicken liver (or lentil+spinach for veg) + tomato salad + lemon", "Dinner": "Lentil soup + spinach curry + roti + vitamin C salad", "Snack": "Blackstrap molasses in warm water + pumpkin seeds + dates"}, "nutrition": ["Iron 30-45mg/day (oral supplements likely needed)", "Vitamin C 1000mg/day", "B12 daily", "Folate 800mcg/day"], "hydration": "12 glasses water. Avoid tea/coffee entirely for 2-3 months. Nettle + hibiscus tea are iron-supportive.", "supplements": ["See doctor — oral iron supplements likely needed", "Vitamin B12 1000-5000mcg (injection may be needed)", "Folate 1mg", "Vitamin C 1000mg", "Vitamin D3"]},
        },
    }

    # Fallback generic diet
    fallback = {
        "headline"   : "Balanced Healthy Diet 🥗",
        "foods_eat"  : ["🥬 Plenty of vegetables (half your plate)", "🌾 Whole grains", "🥩 Lean protein", "🥑 Healthy fats in moderation", "🫐 Fruits in moderation"],
        "foods_avoid": ["❌ Processed and ultra-processed foods", "❌ Added sugars", "❌ Excess sodium", "❌ Alcohol"],
        "meal_plan"  : {"Breakfast": "Oats + fruits + nuts", "Lunch": "Rice/roti + dal + vegetables + salad", "Dinner": "Protein + vegetables + whole grain", "Snack": "Nuts or fruits"},
        "nutrition"  : ["Balanced macros", "Plenty of fibre", "Micronutrient-dense foods"],
        "hydration"  : "Drink 8-10 glasses of water daily.",
        "supplements": ["Multivitamin", "Vitamin D3", "Omega-3"],
    }

    return diets.get(disease_key, {}).get(risk_level, fallback)

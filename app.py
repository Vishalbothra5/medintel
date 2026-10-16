"""
app.py — Medical AI Diagnosis System
=====================================
Layer 1 : Rule-Based Engine  (keyword matching against prescription DB)
Layer 2 : XGBoost Classifier (ML model on 132 symptom columns)
Layer 3 : AI Agent           (OpenRouter LLM with fallback chain)

Decision Flow:
  ┌─────────────────────────────────────┐
  │  Layer 1 + Layer 2 run in parallel  │
  │  Winner = highest confidence        │
  └──────────────────┬──────────────────┘
                     │
         ┌───────────▼───────────┐
         │  Disease identified?  │
         └───────┬───────┬───────┘
                YES      NO
                 │        │
    ┌────────────▼─┐   ┌──▼──────────────────────┐
    │ DB lookup    │   │ Layer 3 Agent predicts   │
    └──────┬───────┘   │ disease + precautions    │
           │           └──────────────────────────┘
    ┌──────▼──────┐
    │ DB record?  │
    └──┬──────┬───┘
      YES     NO
       │       │
  Return    ┌──▼──────────────────────┐
  DB data   │ Layer 3 Agent provides  │
  + Agent   │ full precautions for    │
  enhances  │ unknown disease         │
            └─────────────────────────┘
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from agent import agent_predict_disease, agent_no_db_precautions, agent_enhance_precautions

app = Flask(__name__)

# ── Load Prescription DB ────────────────────────────────────
DB = pd.read_csv("prescription_db.csv")
DB["disease_lower"] = DB["disease"].str.lower().str.strip()

# ── Load XGBoost models ─────────────────────────────────────
XGB_LOADED = False
try:
    xgb_model       = joblib.load("xgb_disease_model.pkl")
    label_encoder   = joblib.load("label_encoder.pkl")
    symptom_columns = joblib.load("symptom_columns.pkl")
    XGB_LOADED = True
    print("[OK] XGBoost model loaded.")
except Exception as e:
    print(f"[WARN] XGBoost not loaded: {e}")

# ── Allergy drug map ────────────────────────────────────────
ALLERGY_MAP = {
    "penicillin":     ["penicillin", "amoxicillin", "ampicillin", "augmentin"],
    "aspirin":        ["aspirin", "ibuprofen", "diclofenac", "naproxen", "nsaid"],
    "sulfa":          ["sulfa", "sulfamethoxazole", "cotrimoxazole", "bactrim"],
    "ibuprofen":      ["ibuprofen", "brufen", "advil", "naproxen"],
    "metformin":      ["metformin", "glucophage"],
    "codeine":        ["codeine", "morphine", "tramadol"],
    "amoxicillin":    ["amoxicillin", "augmentin", "ampicillin"],
    "nitrofurantoin": ["nitrofurantoin"],
    "ciprofloxacin":  ["ciprofloxacin"],
    "azithromycin":   ["azithromycin"],
}


# ════════════════════════════════════════════════════════════
#  LAYER 1: Rule-Based
# ════════════════════════════════════════════════════════════
def rule_based_predict(symptoms_input: str):
    symptoms_lower = symptoms_input.lower()
    best_disease = None
    best_score   = 0
    best_conf    = 0
    top5         = []

    for _, row in DB.iterrows():
        keywords = [k.strip() for k in str(row["symptoms"]).split(",")]
        matched  = sum(1 for kw in keywords if kw in symptoms_lower)
        if matched == 0:
            continue
        score = matched
        conf  = round((matched / len(keywords)) * 100)
        top5.append((row["disease"], score, conf))
        if score > best_score:
            best_score   = score
            best_conf    = conf
            best_disease = row["disease"]

    print("\n" + "=" * 55)
    print("  [LAYER 1] RULE-BASED ENGINE")
    print("=" * 55)
    if top5:
        for d, s, c in sorted(top5, key=lambda x: x[1], reverse=True)[:5]:
            marker = " <-- WINNER" if d == best_disease else ""
            print(f"  {d:<30} | hits={s}  conf={c}%{marker}")
    else:
        print("  No matching disease found.")
    print(f"  >> Disease: {best_disease}  Conf: {best_conf}%")
    print("=" * 55)

    return best_disease, best_conf


# ════════════════════════════════════════════════════════════
#  LAYER 2: XGBoost
# ════════════════════════════════════════════════════════════
def xgb_predict(symptoms_list: list):
    if not XGB_LOADED:
        print("\n" + "=" * 55)
        print("  [LAYER 2] XGBOOST — NOT LOADED")
        print("=" * 55)
        return None, 0

    input_df = pd.DataFrame(
        np.zeros((1, len(symptom_columns)), dtype="uint8"),
        columns=symptom_columns,
    )
    matched_cols = []
    for s in symptoms_list:
        s = s.strip().lower()
        if s in input_df.columns:
            input_df.loc[0, s] = 1
            matched_cols.append(s)

    probs    = xgb_model.predict_proba(input_df)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    disease  = label_encoder.classes_[top5_idx[0]]
    conf     = round(float(probs[top5_idx[0]]) * 100, 1)

    print("\n" + "=" * 55)
    print("  [LAYER 2] XGBOOST ENGINE")
    print("=" * 55)
    print(f"  Matched columns: {matched_cols if matched_cols else 'NONE'}")
    for i in top5_idx:
        marker = " <-- WINNER" if i == top5_idx[0] else ""
        print(f"  {label_encoder.classes_[i]:<30} | {round(float(probs[i])*100,1)}%{marker}")
    print(f"  >> Disease: {disease}  Conf: {conf}%")
    print("=" * 55)

    return disease, conf


# ════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════
def check_allergies(drug_text: str, allergies_str: str) -> list:
    warnings = []
    if not allergies_str or allergies_str.lower() in ["none", "no", ""]:
        return warnings
    drug_lower = drug_text.lower()
    for a in allergies_str.split(","):
        a     = a.strip().lower()
        drugs = ALLERGY_MAP.get(a, [a])
        for d in drugs:
            if d in drug_lower:
                warnings.append(f"{a.upper()} allergy — AVOID {d} found in prescription!")
    return warnings


def get_prescription(disease_name: str, age: str):
    match = DB[DB["disease_lower"] == disease_name.lower().strip()]
    if match.empty:
        match = DB[DB["disease_lower"].str.contains(disease_name.lower().strip(), na=False)]
    if match.empty:
        return None
    row = match.iloc[0]
    try:
        age_int = int(age)
    except Exception:
        age_int = 99
    dosage = row["adult_dosage"] if age_int >= 18 else row["child_dosage"]
    return {
        "disease":       row["disease"],
        "category":      row["category"],
        "drug_name":     row["drug_name"],
        "dosage":        dosage,
        "frequency":     row["frequency"],
        "duration_days": row["duration_days"],
        "what_to_avoid": row["what_to_avoid"],
        "precautions":   row["precautions"],
        "emergency":     row["emergency_flag"],
    }


# ════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════
@app.route("/")
def index():
    all_symptoms = sorted(set(
        s.strip()
        for symptoms in DB["symptoms"].dropna()
        for s in symptoms.split(",")
    ))
    return render_template("index.html", all_symptoms=all_symptoms)


@app.route("/predict", methods=["POST"])
def predict():
    data            = request.json
    symptoms_text   = data.get("symptoms", "").strip()
    symptoms_list   = [s.strip() for s in symptoms_text.split(",") if s.strip()]
    age             = data.get("age", "25")
    gender          = data.get("gender", "")
    allergies       = data.get("allergies", "")
    medical_history = data.get("medical_history", "")

    if not symptoms_text:
        return jsonify({"error": "Please enter at least one symptom."}), 400

    # ── Request log ─────────────────────────────────────────
    print("\n" + "#" * 55)
    print("  [NEW PREDICTION REQUEST]")
    print("#" * 55)
    print(f"  Symptoms : {symptoms_text}")
    print(f"  Age      : {age}  | Gender: {gender}")
    print(f"  Allergies: {allergies or 'None'}")
    print(f"  History  : {medical_history or 'None'}")
    print("#" * 55)

    # ── Layer 1 + 2 ─────────────────────────────────────────
    rule_disease, rule_conf = rule_based_predict(symptoms_text)
    xgb_disease,  xgb_conf  = xgb_predict(symptoms_list)

    # ── Decision ─────────────────────────────────────────────
    print("\n" + "-" * 55)
    print("  [DECISION]")
    print("-" * 55)
    print(f"  Layer 1 Rule-Based => {rule_disease} ({rule_conf}%)")
    print(f"  Layer 2 XGBoost    => {xgb_disease} ({xgb_conf}%)")

    if rule_disease and xgb_disease:
        if rule_conf >= xgb_conf:
            final_disease, final_conf, winner_src = rule_disease, rule_conf, "Rule-Based"
        else:
            final_disease, final_conf, winner_src = xgb_disease, xgb_conf, "XGBoost"
    elif rule_disease:
        final_disease, final_conf, winner_src = rule_disease, rule_conf, "Rule-Based"
    elif xgb_disease:
        final_disease, final_conf, winner_src = xgb_disease, xgb_conf, "XGBoost"
    else:
        final_disease, final_conf, winner_src = None, 0, None

    # ── Layer 3 triggered? ───────────────────────────────────
    agent_data      = None
    layer3_triggered = False
    trigger_reason  = ""

    if not final_disease:
        # Both layers failed — agent predicts + precautions
        trigger_reason   = "both_layers_failed"
        layer3_triggered = True
        print("  >> Layer 1 & 2 FAILED — triggering Layer 3 Agent (predict mode)")
        print("-" * 55)
        agent_data = agent_predict_disease(
            symptoms_text, age, gender, allergies, medical_history
        )
        if "error" in agent_data:
            return jsonify({"error": "All three layers failed. " + agent_data["error"]}), 200

        # Use agent's prediction as final
        final_disease = agent_data.get("disease", "Unknown")
        final_conf    = agent_data.get("confidence", 0)
        winner_src    = f"AI Agent ({agent_data.get('model_used', 'unknown')})"
    else:
        print(f"  >> WINNER => {final_disease} ({final_conf}%) [{winner_src}]")

    print("-" * 55)

    # ── DB Lookup ────────────────────────────────────────────
    prescription = get_prescription(final_disease, age) if final_disease else None

    if not prescription and not layer3_triggered:
        # Disease found but NO DB record — agent fills the gap
        trigger_reason   = "no_db_record"
        layer3_triggered = True
        print(f"  >> No DB record for '{final_disease}' — triggering Layer 3 (precaution mode)")
        agent_data = agent_no_db_precautions(
            final_disease, age, gender, allergies, medical_history, symptoms_text
        )

    # ── Build response ───────────────────────────────────────
    allergy_warnings = []

    if prescription:
        drug_text = prescription["drug_name"] + " " + prescription["dosage"]
        allergy_warnings = check_allergies(drug_text, allergies)

        # Even with DB record, run agent to enhance precautions (non-blocking)
        if not layer3_triggered:
            try:
                agent_data = agent_enhance_precautions(
                    disease=final_disease,
                    existing_precautions=prescription["precautions"],
                    age=age,
                    gender=gender,
                    allergies=allergies,
                    medical_history=medical_history,
                    symptoms_text=symptoms_text,
                )
                layer3_triggered = True
                trigger_reason   = "precaution_enhancement"
            except Exception as e:
                print(f"  [AGENT] Enhancement skipped: {e}")

        response = {
            # Core identification
            "disease":          prescription["disease"],
            "category":         prescription["category"],
            "confidence":       final_conf,
            "winner_source":    winner_src,
            "age_group":        "Adult (18+)" if int(age) >= 18 else "Child (<18)",

            # Prescription from DB
            "drug_name":        prescription["drug_name"],
            "dosage":           prescription["dosage"],
            "frequency":        prescription["frequency"],
            "duration_days":    str(prescription["duration_days"]),
            "what_to_avoid":    prescription["what_to_avoid"],
            "precautions":      prescription["precautions"],
            "emergency":        str(prescription["emergency"]),
            "allergy_warnings": allergy_warnings,

            # Layer info
            "layers_used":      {"layer1": bool(rule_disease), "layer2": bool(xgb_disease), "layer3": layer3_triggered},
            "layer3_trigger":   trigger_reason,
        }

        # Merge agent data if available
        if agent_data and "error" not in agent_data:
            response["agent"] = {
                "enhanced_precautions": agent_data.get("enhanced_precautions", ""),
                "lifestyle_tips":       agent_data.get("lifestyle_tips", ""),
                "warning_signs":        agent_data.get("warning_signs", ""),
                "follow_up":            agent_data.get("follow_up", ""),
                "allergy_warnings_ai":  agent_data.get("allergy_warnings", ""),
                "agent_note":           agent_data.get("agent_note", ""),
                "model_used":           agent_data.get("model_used", ""),
                "source":               agent_data.get("source", ""),
            }

    else:
        # Full agent response (no DB record at all)
        if agent_data and "error" not in agent_data:
            allergy_warnings = check_allergies(
                agent_data.get("drug_suggestion", ""), allergies
            )
            response = {
                "disease":          final_disease,
                "category":         agent_data.get("category", "AI Assessed"),
                "confidence":       final_conf,
                "winner_source":    winner_src,
                "age_group":        "Adult (18+)" if int(age) >= 18 else "Child (<18)",

                # From agent
                "drug_name":        agent_data.get("drug_suggestion", "Consult a doctor"),
                "dosage":           agent_data.get("dosage_note", "As prescribed"),
                "frequency":        agent_data.get("frequency", "As prescribed"),
                "duration_days":    agent_data.get("duration_note", "As prescribed"),
                "what_to_avoid":    agent_data.get("what_to_avoid", ""),
                "precautions":      agent_data.get("precautions", agent_data.get("enhanced_precautions", "")),
                "emergency":        agent_data.get("emergency", "No"),
                "allergy_warnings": allergy_warnings,

                "layers_used":      {"layer1": bool(rule_disease), "layer2": bool(xgb_disease), "layer3": True},
                "layer3_trigger":   trigger_reason,

                "agent": {
                    "enhanced_precautions": agent_data.get("enhanced_precautions", agent_data.get("precautions", "")),
                    "lifestyle_tips":       agent_data.get("lifestyle_tips", ""),
                    "warning_signs":        agent_data.get("warning_signs", ""),
                    "follow_up":            agent_data.get("follow_up", ""),
                    "emergency_reason":     agent_data.get("emergency_reason", ""),
                    "allergy_warnings_ai":  agent_data.get("allergy_warnings", ""),
                    "agent_note":           agent_data.get("agent_note", ""),
                    "model_used":           agent_data.get("model_used", ""),
                    "source":               agent_data.get("source", ""),
                },
            }
        else:
            return jsonify({
                "error": f"Disease '{final_disease}' identified but no prescription found and AI agent unavailable.",
                "disease": final_disease,
                "winner_source": winner_src,
                "confidence": final_conf,
            }), 200

    # ── Final log ────────────────────────────────────────────
    print("\n" + "-" * 55)
    print("  [PRESCRIPTION ISSUED]")
    print("-" * 55)
    print(f"  Disease  : {response['disease']}")
    print(f"  Drug     : {response['drug_name']}")
    print(f"  Dosage   : {response['dosage']}")
    print(f"  Source   : {response['winner_source']}")
    print(f"  Layer 3  : {'YES — ' + trigger_reason if layer3_triggered else 'Not triggered'}")
    if allergy_warnings:
        for w in allergy_warnings:
            print(f"  !! ALLERGY: {w}")
    print("-" * 55)

    return jsonify(response)


@app.route("/diseases")
def diseases():
    return jsonify(sorted(DB["disease"].tolist()))


@app.route("/status")
def status():
    """Health check endpoint."""
    import os
    api_key_set = bool(os.getenv("OPENROUTER_API_KEY"))
    return jsonify({
        "xgboost_loaded": XGB_LOADED,
        "db_diseases": len(DB),
        "agent_ready": api_key_set,
        "fallback_models": ["google/gemini-2.0-flash-001", "anthropic/claude-3-haiku",
                            "meta-llama/llama-3.3-70b-instruct", "mistralai/mistral-7b-instruct"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)

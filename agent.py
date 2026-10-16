"""
agent.py — Layer 3: AI Medical Agent (OpenRouter + Fallback LLMs)

Triggered when:
  1. Both Rule-Based + XGBoost fail to identify a disease
  2. Disease found but NOT in the prescription DB (no precautions available)
  3. As an enhancer to add richer precautions for any diagnosis

Fallback chain (best → reliable → lightweight):
  1. google/gemini-2.0-flash-001         (fast, capable)
  2. anthropic/claude-3-haiku             (concise medical reasoning)
  3. meta-llama/llama-3.3-70b-instruct   (open-weights, strong)
  4. mistralai/mistral-7b-instruct       (last resort lightweight)
"""

import os
import json
import requests
import time
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Fallback model chain ─────────────────────────────────────
FALLBACK_MODELS = [
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-7b-instruct",
]

HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5000",
    "X-Title": "MedicalAI-Layer3",
}


def _build_headers():
    h = HEADERS.copy()
    if OPENROUTER_API_KEY:
        h["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
    return h


def _call_openrouter(model: str, messages: list, max_tokens: int = 900) -> str | None:
    """Call a single OpenRouter model. Returns text or None on failure."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers=_build_headers(),
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            return content
        else:
            print(f"  [AGENT] Model {model} returned HTTP {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  [AGENT] Model {model} exception: {e}")
        return None


def _call_with_fallback(messages: list, max_tokens: int = 900) -> tuple[str | None, str | None]:
    """
    Try each model in FALLBACK_MODELS.
    Returns (response_text, model_used) or (None, None) if all fail.
    """
    for model in FALLBACK_MODELS:
        print(f"  [AGENT] Trying model: {model}")
        result = _call_openrouter(model, messages, max_tokens)
        if result:
            print(f"  [AGENT] Success with: {model}")
            return result, model
        time.sleep(0.5)  # brief pause before next fallback
    return None, None


def _parse_json_response(text: str) -> dict | None:
    """Safely extract JSON from LLM response."""
    try:
        # Try direct parse first
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to extract from markdown code block
    for marker in ["```json", "```"]:
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
    return None


# ════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════

def agent_predict_disease(
    symptoms_text: str,
    age: str,
    gender: str,
    allergies: str,
    medical_history: str,
) -> dict:
    """
    Layer 3 — FULL agent: predict disease + give precautions.
    Called when both Rule-Based and XGBoost fail.

    Returns a dict with keys:
        disease, confidence, category, drug_suggestion,
        dosage_note, precautions, what_to_avoid, emergency,
        agent_note, model_used, source
    """
    print("\n" + "=" * 55)
    print("  [LAYER 3 AGENT] — Disease Prediction Mode")
    print("=" * 55)

    system_prompt = """You are an expert medical AI assistant. 
Your job is to analyze patient symptoms and provide a structured medical assessment.
You must respond ONLY with valid JSON — no prose, no markdown, no explanation outside the JSON.
Always include a disclaimer that your output is AI-generated and the patient should consult a doctor."""

    user_prompt = f"""A patient has the following details:
- Symptoms: {symptoms_text}
- Age: {age}
- Gender: {gender}
- Known Allergies: {allergies if allergies else "None"}
- Medical History: {medical_history if medical_history else "None"}

Based on the symptoms, analyze and return a JSON object with these exact keys:
{{
  "disease": "<most likely disease name>",
  "confidence": <integer 0-100>,
  "category": "<disease category e.g. Infectious, Cardiovascular, Neurological>",
  "drug_suggestion": "<recommended drug(s) — general name, not brand>",
  "dosage_note": "<general dosage guidance for age {age}>",
  "frequency": "<e.g. twice daily, every 8 hours>",
  "duration_note": "<e.g. 5-7 days or as prescribed>",
  "precautions": "<3-5 specific precautions the patient should follow>",
  "what_to_avoid": "<food, activities, or drugs to avoid>",
  "emergency": "<Yes or No — should patient go to ER immediately?>",
  "emergency_reason": "<if Yes, explain why>",
  "allergy_warnings": "<any warnings based on known allergies, or 'None'>",
  "agent_note": "AI-generated assessment. This is not a substitute for professional medical advice. Please consult a licensed doctor."
}}

Be clinically accurate. If the symptoms are ambiguous, state the most probable disease and note alternatives in precautions."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw, model_used = _call_with_fallback(messages, max_tokens=900)

    if not raw:
        return {
            "error": "AI Agent is currently unavailable. All fallback models failed.",
            "source": "agent_failed",
        }

    parsed = _parse_json_response(raw)
    if not parsed:
        # Return raw text wrapped if JSON parse fails
        return {
            "disease": "Unknown (AI parse error)",
            "confidence": 0,
            "precautions": raw[:500],
            "agent_note": "AI response could not be structured. Raw text provided.",
            "model_used": model_used,
            "source": "agent_layer3",
        }

    parsed["model_used"] = model_used
    parsed["source"] = "agent_layer3"
    print(f"  [AGENT] Predicted: {parsed.get('disease')}  ({parsed.get('confidence')}%)")
    print("=" * 55)
    return parsed


def agent_enhance_precautions(
    disease: str,
    existing_precautions: str,
    age: str,
    gender: str,
    allergies: str,
    medical_history: str,
    symptoms_text: str,
) -> dict:
    """
    Layer 3 — PRECAUTION ENHANCER mode.
    Called when disease is found but NOT in DB (no prescription record),
    or to enrich existing DB precautions with AI advice.

    Returns a dict with keys:
        enhanced_precautions, lifestyle_tips, warning_signs,
        follow_up, agent_note, model_used, source
    """
    print("\n" + "=" * 55)
    print("  [LAYER 3 AGENT] — Precaution Enhancement Mode")
    print(f"  Disease: {disease}")
    print("=" * 55)

    system_prompt = """You are a medical AI providing detailed, patient-friendly precautions.
Respond ONLY with valid JSON. No markdown, no prose outside JSON."""

    user_prompt = f"""Patient diagnosed with: {disease}
Patient details:
- Symptoms: {symptoms_text}
- Age: {age}, Gender: {gender}
- Allergies: {allergies if allergies else "None"}
- Medical History: {medical_history if medical_history else "None"}
- Existing precautions from database: {existing_precautions if existing_precautions else "None available"}

Provide a comprehensive medical guidance JSON:
{{
  "drug_suggestion": "<recommended OTC or prescription drug(s) if disease not treated>",
  "dosage_note": "<appropriate dosage for age {age}>",
  "frequency": "<administration frequency>",
  "duration_note": "<typical treatment duration>",
  "enhanced_precautions": "<detailed 4-6 point precautions specific to {disease}>",
  "lifestyle_tips": "<diet, rest, hydration, and activity modifications>",
  "warning_signs": "<symptoms that indicate worsening — patient must see doctor immediately>",
  "what_to_avoid": "<specific foods, activities, or medications to avoid for {disease}>",
  "follow_up": "<when and how patient should follow up with doctor>",
  "emergency": "<Yes or No>",
  "allergy_warnings": "<drug allergy warnings based on {allergies}>",
  "agent_note": "AI-generated guidance. Always consult a licensed medical professional before starting any treatment."
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw, model_used = _call_with_fallback(messages, max_tokens=900)

    if not raw:
        return {
            "error": "AI Agent unavailable for precaution enhancement.",
            "source": "agent_failed",
        }

    parsed = _parse_json_response(raw)
    if not parsed:
        return {
            "enhanced_precautions": raw[:600],
            "agent_note": "AI response partially parsed.",
            "model_used": model_used,
            "source": "agent_layer3_precautions",
        }

    parsed["model_used"] = model_used
    parsed["source"] = "agent_layer3_precautions"
    print(f"  [AGENT] Precautions enhanced for: {disease}")
    print("=" * 55)
    return parsed


def agent_no_db_precautions(
    disease: str,
    age: str,
    gender: str,
    allergies: str,
    medical_history: str,
    symptoms_text: str,
) -> dict:
    """
    Shorthand: disease known but zero DB record.
    Delegates to enhance with empty existing_precautions.
    """
    return agent_enhance_precautions(
        disease=disease,
        existing_precautions="",
        age=age,
        gender=gender,
        allergies=allergies,
        medical_history=medical_history,
        symptoms_text=symptoms_text,
    )

"""
SmartHealth Community AI — Layer 4: Central LLM Gateway & Station 3 Worker
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: smarthealth-community-ai/llm_gateway.py
Responsibility: Queries Code for Africa's LiteLLM Gateway with Doctor Guardrails, 
translates the clinical response to the target dialect (Pidgin or Hausa), and deposits into `q_tts`.
"""

import os
import asyncio
from openai import OpenAI
from core_engine import q_llm, q_tts, collect_batch
from translator_model import translate_from_english
from config.loader import API, MODELS, PARAMS, QUEUES

# =============================================================================
# 0. LITELLM GATEWAY & BEDROCK CONFIGURATION
# =============================================================================
# Use env vars for secrets, fallback to config for structural defaults
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", API["litellm_base_url"])
TEAM_KEY = os.getenv("TEAM_KEY", "your_cfa_team_key_here")
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", MODELS["llm_default"])

print(f"[SmartHealth LLM] Initializing Gateway Client to: {LITELLM_BASE_URL}")
client = OpenAI(
    base_url=LITELLM_BASE_URL,
    api_key=TEAM_KEY
)

# =============================================================================
# 1. DOCTOR-VALIDATED CLINICAL TRIAGE SYSTEM PROMPT
# =============================================================================
DOCTOR_TRIAGE_SYSTEM_PROMPT = """You are an expert, empathetic clinical triage assistant for Nigerian community healthcare (WOGAFH SmartHealth AI).

CRITICAL CLINICAL RULES FROM OUR MEDICAL OFFICER:
1. TRIAGE FIRST, PRESCRIBE NEVER: Never prescribe systemic prescription antibiotics (e.g. Ceftriaxone, Doxycycline, Ciprofloxacin) or injections over chat.
2. FIRST AID ONLY: Provide safe, over-the-counter first aid only (e.g. Paracetamol for pain, ORS for dehydration, shock positioning by elevating legs).
3. MANDATORY FACILITY REFERRAL: Always instruct the patient or Traditional Birth Attendant to proceed immediately to the nearest Primary Health Centre (PHC) or General Hospital for physical examination, lab swabs, and ultrasound.
4. BEDSIDE EMPATHY & CALM: Maintain a reassuring, supportive tone to prevent unnecessary patient panic while clearly explaining red-flag symptoms.
5. STRICT GROUNDING: Anchor all medical advice strictly in the provided Nigerian Standard Treatment Guidelines excerpts.
6. NO HALLUCINATION: If the provided guidelines do not directly address the patient's symptom, explicitly state: "This specific symptom requires an in-person evaluation by a healthcare professional. Please proceed to the nearest clinic." Do not guess or invent medical advice.
"""

# =============================================================================
# 2. PROMPT BUILDER & GATEWAY INFERENCE HELPER
# =============================================================================
def query_clinical_llm(patient_text: str, guideline_chunks: list[dict]) -> str:
    """
    Constructs the prompt combining the patient's symptom and the retrieved Top-K STG chunks,
    then calls LiteLLM Gateway for clinical reasoning in standard clinical English.
    """
    context_block = ""
    for i, chunk in enumerate(guideline_chunks, start=1):
        context_block += f"\n[Reference {i}]\n"
        context_block += f"Source: {chunk.get('source', 'FMoH Guidelines')} (Page {chunk.get('page', 'Unknown')})\n"
        context_block += f"Protocol: \"{chunk.get('text', '')}\"\n"

    user_prompt = f"""PATIENT SYMPTOM / QUERY:
"{patient_text}"

OFFICIAL NIGERIAN STANDARD TREATMENT GUIDELINE REFERENCES:
{context_block.strip()}

Provide a clear, safe, step-by-step clinical triage and first-aid response in clear English based *only* on the references above."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": DOCTOR_TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=PARAMS["llm_temperature"],
            max_tokens=PARAMS["llm_max_tokens"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[SmartHealth LLM Warning] Gateway call failed: {e}. Applying offline fallback...")
        return (
            "This is a serious medical situation. Do not take self-prescribed medications at home. "
            "Please proceed immediately to the nearest Primary Health Centre or General Hospital for evaluation."
        )

# =============================================================================
# 3. STATION 3 ASYNC WORKER LOOP (The Cognitive Core & Translation Bridge)
# =============================================================================
async def llm_gateway_worker():
    """
    Station 3 Background Worker.
    - Listens to: `q_llm`
    - Trigger Condition: N items OR T ms timeout (from config)
    - Action: Executes clinical reasoning, translates verified advice, deposits to `q_tts`.
    """
    max_batch_size = QUEUES["llm"]["max_batch_size"]
    timeout_ms = QUEUES["llm"]["timeout_ms"]
    print(f"[Worker 3: LLM Gateway Active] Connected to {LITELLM_BASE_URL} (N={max_batch_size}, T={timeout_ms}ms)")
    
    while True:
        batch = await collect_batch(q_llm, max_batch_size, timeout_ms)
        
        for item in batch:
            try:
                transcript = item["transcription"]
                guidelines = item["matched_guidelines"] 
                dialect = item["detected_dialect"]
                
                # Step 1: Execute Reasoning via Gateway
                english_advice = query_clinical_llm(transcript, guidelines)
                
                # Step 2: Translate advice into the patient's dialect
                final_speech_text = translate_from_english(english_advice, dialect)
                
                # Step 3: Populate packet with speech-ready text
                item["clinical_advice"] = final_speech_text
                
                # Step 4: HAND OFF TO STATION 4 (TTS Speech Synthesis Queue)
                await q_tts.put(item)
                
            except Exception as e:
                item["future"].set_exception(e)
            finally:
                q_llm.task_done()

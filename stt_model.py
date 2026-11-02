"""
SmartHealth Community AI — Layer 1: Speech-to-Text (STT) & Language ID
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: smarthealth-community-ai/stt_model.py
Responsibility: Detects language (Hausa/Pidgin) and transcribes 16kHz audio 
using Meta MMS-1B, optimized for sub-50ms inference per request.
"""

import asyncio
import torch
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    AutoProcessor,
    Wav2Vec2ForCTC
)
from core_engine import q_stt, q_rag, collect_batch
from config.loader import MODELS, API, QUEUES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SmartHealth Models] Initializing STT/LID models on hardware: {DEVICE.upper()}")

# =============================================================================
# 1. LOAD LANGUAGE IDENTIFICATION (LID) MODEL
# =============================================================================
stt_lid_model_name = MODELS["stt_lid"]
print(f"[1/2 Loading] Meta MMS-LID (`{stt_lid_model_name}`)...")
lid_processor = AutoFeatureExtractor.from_pretrained(stt_lid_model_name)
lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(stt_lid_model_name).to(DEVICE).eval()

# =============================================================================
# 2. LOAD MMS-1B STT & PRE-LOAD ADAPTERS (Prevents live request timeouts)
# =============================================================================
stt_base_model_name = MODELS["stt_base"]
print(f"[2/2 Loading] Meta MMS-1B STT (`{stt_base_model_name}`)...")
stt_processor = AutoProcessor.from_pretrained(stt_base_model_name)
stt_model = Wav2Vec2ForCTC.from_pretrained(stt_base_model_name).to(DEVICE).eval()

stt_adapters = MODELS["stt_adapters"]
print(f" -> Pre-caching {', '.join(stt_adapters)} adapters into RAM...")
for adapter in stt_adapters:
    stt_model.load_adapter(adapter)
print("✓ STT and Language Adapters pre-cached successfully in RAM!\n")

# =============================================================================
# 3. INFERENCE HELPERS
# =============================================================================
def detect_language(audio_16k_array) -> str:
    """
    Analyzes the first 3 seconds (48,000 samples) of audio to detect 
    if the speaker is using Hausa or Nigerian Pidgin.
    """
    sample_window = audio_16k_array[:48000]
    inputs = lid_processor(sample_window, sampling_rate=API["target_sample_rate"], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        logits = lid_model(**inputs).logits
        
    pred_id = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
    detected_tag = lid_model.config.id2label[pred_id].lower()
    
    # Fallback to pcm if hau is not explicitly detected
    return "hau" if "hau" in detected_tag else "pcm"

def transcribe_mms(audio_16k_array, dialect_code: str) -> str:
    """
    Transcribes audio using the pre-loaded MMS-1B model.
    """
    stt_processor.tokenizer.set_target_lang(dialect_code)
    inputs = stt_processor(audio_16k_array, sampling_rate=API["target_sample_rate"], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        logits = stt_model(inputs.input_values).logits
        
    predicted_ids = torch.argmax(logits, dim=-1)
    return stt_processor.decode(predicted_ids[0]).strip()

# =============================================================================
# 4. STATION 1 ASYNC WORKER LOOP
# =============================================================================
async def stt_model_worker():
    """
    Station 1 Background Worker.
    - Listens to: `q_stt`
    - Trigger Condition: N items OR T ms timeout (from config)
    - Action: Detects language, transcribes audio, and deposits to `q_rag`.
    """
    max_batch_size = QUEUES["stt"]["max_batch_size"]
    timeout_ms = QUEUES["stt"]["timeout_ms"]
    print(f"[Worker 1: STT/LID Active] Ready on {DEVICE.upper()} (N={max_batch_size}, T={timeout_ms}ms)")
    
    while True:
        batch = await collect_batch(q_stt, max_batch_size, timeout_ms)
        
        for item in batch:
            try:
                raw_audio = item["audio_16k"]
                
                # Step 1: Detect Language
                dialect = detect_language(raw_audio)
                
                # Step 2: Transcribe
                transcription = transcribe_mms(raw_audio, dialect)
                
                # Step 3: Populate packet
                item["detected_dialect"] = dialect
                item["transcription"] = transcription
                
                # Step 4: HAND OFF TO STATION 2 (RAG/Safety Queue)
                await q_rag.put(item)
                
            except Exception as e:
                item["future"].set_exception(e)
            finally:
                q_stt.task_done()

"""
SmartHealth Community AI — Layer 2: Speech Models & Station 1 (STT / LID)
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: stt_model.py
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SmartHealth Models] Initializing models on hardware: {DEVICE.upper()}")

# 1. Load LID Model
print("[1/2 Loading] Meta MMS-LID (`facebook/mms-lid-126`)...")
lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-126")
lid_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-126").to(DEVICE).eval()

# 2. Load MMS-1B STT & Pre-load Adapters (Prevents live request timeouts)
print("[2/2 Loading] Meta MMS-1B STT (`facebook/mms-1b-all`)...")
stt_processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(DEVICE).eval()

print(" -> Pre-caching Hausa (`hau`) & Pidgin (`pcm`) adapters...")
stt_model.load_adapter("hau")
stt_model.load_adapter("pcm")
print("✓ STT and Language Adapters pre-cached successfully in RAM!\n")

def detect_language(audio_16k_array) -> str:
    sample_window = audio_16k_array[:48000]
    inputs = lid_processor(sample_window, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = lid_model(**inputs).logits
    pred_id = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
    detected_tag = lid_model.config.id2label[pred_id].lower()
    return "hau" if "hau" in detected_tag else "pcm"

def transcribe_mms(audio_16k_array, dialect_code: str) -> str:
    stt_processor.tokenizer.set_target_lang(dialect_code)
    stt_model.load_adapter(dialect_code)
    inputs = stt_processor(audio_16k_array, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = stt_model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return stt_processor.decode(predicted_ids[0]).strip()

async def stt_model_worker(max_batch_size: int = 4, timeout_ms: int = 50):
    print(f"[Worker 1: STT/LID Active] Ready on {DEVICE.upper()} (N={max_batch_size}, T={timeout_ms}ms)")
    while True:
        batch = await collect_batch(q_stt, max_batch_size, timeout_ms)
        for item in batch:
            try:
                raw_audio = item["audio_16k"]
                dialect = detect_language(raw_audio)
                transcription = transcribe_mms(raw_audio, dialect)
                item["detected_dialect"] = dialect
                item["transcription"] = transcription
                await q_rag.put(item)
            except Exception as e:
                item["future"].set_exception(e)
            finally:
                q_stt.task_done()

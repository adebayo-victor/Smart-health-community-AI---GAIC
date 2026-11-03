"""
SmartHealth Community AI — Layer 5: Text-to-Speech (TTS) & Station 4 Worker
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: smarthealth-community-ai/tts_model.py
Responsibility: Loads Meta MMS-TTS VITS models for Hausa and Nigerian Pidgin,
synthesizes 16kHz lossless WAV audio voice notes in <0.35s, and resolves the client's final Future.
"""

import os
import time
import torch
import soundfile as sf
from transformers import AutoTokenizer, VitsModel
from core_engine import q_tts, collect_batch
from config.loader import PATHS, MODELS, PARAMS, QUEUES

# =============================================================================
# 0. HARDWARE DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_AUDIO_DIR = PATHS["audio_output_dir"]
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

print(f"[SmartHealth TTS] Initializing Speech Synthesis Engine on: {DEVICE.upper()}")

# =============================================================================
# 1. LOAD META MMS-TTS VITS MODELS (Hausa & Nigerian Pidgin)
# =============================================================================
print(f"[1/2 Loading] Meta MMS-TTS Hausa (`{MODELS['tts_hausa']}`)...")
tts_hau_tokenizer = AutoTokenizer.from_pretrained(MODELS["tts_hausa"])
tts_hau_model = VitsModel.from_pretrained(MODELS["tts_hausa"]).to(DEVICE)
tts_hau_model.eval()

print(f"[2/2 Loading] Meta MMS-TTS Nigerian Pidgin (`{MODELS['tts_pidgin']}`)...")
tts_pcm_tokenizer = AutoTokenizer.from_pretrained(MODELS["tts_pidgin"])
tts_pcm_model = VitsModel.from_pretrained(MODELS["tts_pidgin"]).to(DEVICE)
tts_pcm_model.eval()

print("[SmartHealth TTS] Both VITS speech synthesizers loaded successfully into memory!\n")

# =============================================================================
# 2. INFERENCE SYNTHESIS HELPER (Lossless WAV)
# =============================================================================
def synthesize_audio_file(text: str, dialect_code: str, trace_id: str) -> tuple[str, float]:
    if dialect_code == "hau":
        tokenizer = tts_hau_tokenizer
        model = tts_hau_model
    else:
        tokenizer = tts_pcm_tokenizer
        model = tts_pcm_model
        
    output_filename = f"response_{trace_id}.wav"
    output_path = os.path.join(OUTPUT_AUDIO_DIR, output_filename)
    
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    inputs["input_ids"] = inputs["input_ids"].long()
    with torch.no_grad():
        output_waveform = model(**inputs).waveform
        
    audio_array = output_waveform.squeeze().cpu().numpy()
    sampling_rate = model.config.sampling_rate
    audio_duration = len(audio_array) / sampling_rate
    
    # STRICTLY WAV: Using soundfile with PCM_16 (Lossless 16-bit WAV)
    sf.write(output_path, audio_array, sampling_rate, subtype='PCM_16')
    
    return output_path, audio_duration

# =============================================================================
# 3. STATION 4 ASYNC WORKER LOOP
# =============================================================================
async def tts_model_worker():
    max_batch_size = QUEUES["tts"]["max_batch_size"]
    timeout_ms = QUEUES["tts"]["timeout_ms"]
    print(f"[Worker 4: TTS Mouth Active] Ready on {DEVICE.upper()} (N={max_batch_size}, T={timeout_ms}ms)")
    
    while True:
        batch = await collect_batch(q_tts, max_batch_size, timeout_ms)
        
        for item in batch:
            try:
                trace_id = item["trace_id"]
                dialect = item["detected_dialect"]
                advice_text = item["clinical_advice"]
                start_time = item["timestamp_start"]
                
                audio_path, duration = synthesize_audio_file(advice_text, dialect, trace_id)
                item["outbound_audio_path"] = audio_path
                item["audio_duration_seconds"] = round(duration, 2)
                
                total_latency_ms = (time.perf_counter() - start_time) * 1000
                
                item["future"].set_result({
                    "trace_id": trace_id,
                    "dialect": dialect,
                    "transcription": item["transcription"],
                    "is_emergency": item["is_emergency"],
                    "emergency_trigger": item.get("emergency_trigger"),
                    "matched_guidelines": item["matched_guidelines"],
                    "clinical_advice": advice_text,
                    "audio_file_path": audio_path,
                    "audio_duration": item["audio_duration_seconds"],
                    "total_pipeline_latency_ms": round(total_latency_ms, 2),
                    "status": "COMPLETED_SUCCESS"
                })
                
            except Exception as e:
                item["future"].set_exception(e)
            finally:
                q_tts.task_done()

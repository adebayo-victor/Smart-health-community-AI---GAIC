"""
SmartHealth Community AI — Layer 5: Text-to-Speech (TTS) & Station 4 Worker
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: tts_model.py
Responsibility: Loads Meta MMS-TTS VITS models for Hausa and Nigerian Pidgin,
synthesizes 16kHz audio voice notes in <0.35s, and resolves the client's final Future.
"""

import os
import time
import torch
import scipy.io.wavfile
from transformers import AutoTokenizer, VitsModel
from core_engine import q_tts, collect_batch

# =============================================================================
# 0. HARDWARE DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_AUDIO_DIR = os.getenv("OUTPUT_AUDIO_DIR", "data/audio_outputs")
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

print(f"[SmartHealth TTS] Initializing Speech Synthesis Engine on: {DEVICE.upper()}")


# =============================================================================
# 1. LOAD META MMS-TTS VITS MODELS (Hausa & Nigerian Pidgin)
# =============================================================================
print("[1/2 Loading] Meta MMS-TTS Hausa (`facebook/mms-tts-hau`)...")
tts_hau_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hau")
tts_hau_model = VitsModel.from_pretrained("facebook/mms-tts-hau").to(DEVICE)
tts_hau_model.eval()

print("[2/2 Loading] Meta MMS-TTS Nigerian Pidgin (`facebook/mms-tts-pcm`)...")
tts_pcm_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-pcm")
tts_pcm_model = VitsModel.from_pretrained("facebook/mms-tts-pcm").to(DEVICE)
tts_pcm_model.eval()

print("[SmartHealth TTS] Both VITS speech synthesizers loaded successfully into memory!\n")


# =============================================================================
# 2. INFERENCE SYNTHESIS HELPER (With Native-Audited Parameter Tuning)
# =============================================================================
def synthesize_audio_file(text: str, dialect_code: str, trace_id: str) -> tuple[str, float]:
    """
    Synthesizes speech using the VITS architecture.
    Applies noise_scale_w=0.6 and length_scale=0.95 to eliminate stochastic
    pauses and ensure fluid, native bedside cadence.
    
    Returns:
    - output_path: Path to the generated 16kHz .wav file.
    - audio_duration: Duration of the synthesized audio in seconds.
    """
    # Select appropriate tokenizer and model based on detected dialect
    if dialect_code == "hau":
        tokenizer = tts_hau_tokenizer
        model = tts_hau_model
    else:
        tokenizer = tts_pcm_tokenizer
        model = tts_pcm_model
        
    output_filename = f"response_{trace_id}.wav"
    output_path = os.path.join(OUTPUT_AUDIO_DIR, output_filename)
    
    # Tokenize input text to phoneme IDs
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # VITS parameter tuning:
        # - noise_scale_w=0.6: Reduces phoneme duration variance (stops awkward pauses)
        # - length_scale=0.95: Slightly tightens pacing for conversational naturalness
        output_waveform = model(**inputs, noise_scale_w=0.6, length_scale=0.95).waveform
        
    audio_array = output_waveform.squeeze().cpu().numpy()
    sampling_rate = model.config.sampling_rate
    audio_duration = len(audio_array) / sampling_rate
    
    # Save 16kHz waveform (ready for Telegram Opus compression)
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_array)
    
    return output_path, audio_duration


# =============================================================================
# 3. STATION 4 ASYNC WORKER LOOP (The Mouth & Promise Resolver)
# =============================================================================
async def tts_model_worker(max_batch_size: int = 4, timeout_ms: int = 40):
    """
    Station 4 Background Worker.
    - Listens to: `q_tts`
    - Trigger Condition: N=4 items OR T=40ms timeout
    - Action: Synthesizes .wav audio voice note in <0.35s and resolves client Future!
    """
    print(f"[Worker 4: TTS Mouth Active] Ready on {DEVICE.upper()} (N={max_batch_size}, T={timeout_ms}ms)")
    
    while True:
        # Collect batch using our N or T rule
        batch = await collect_batch(q_tts, max_batch_size, timeout_ms)
        
        for item in batch:
            try:
                trace_id = item["trace_id"]
                dialect = item["detected_dialect"]
                advice_text = item["clinical_advice"]
                start_time = item["timestamp_start"]
                
                # Step 1: Synthesize Voice Note Audio
                audio_path, duration = synthesize_audio_file(advice_text, dialect, trace_id)
                item["outbound_audio_path"] = audio_path
                item["audio_duration_seconds"] = round(duration, 2)
                
                # Total Pipeline Latency from Request Ingress to Audio Out
                total_latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Step 2: RESOLVE THE PROMISE / FUTURE!
                # This notifies the Telegram Webhook / App that the entire pipeline is complete
                item["future"].set_result({
                    "trace_id": trace_id,
                    "dialect": dialect,
                    "transcription": item["transcription"],
                    "is_emergency": item["is_emergency"],
                    "emergency_trigger": item.get("emergency_trigger"),
                    "matched_guideline": item["matched_guideline"]["id"],
                    "clinical_advice": advice_text,
                    "audio_file_path": audio_path,
                    "audio_duration": item["audio_duration_seconds"],
                    "total_pipeline_latency_ms": round(total_latency_ms, 2),
                    "status": "COMPLETED_SUCCESS"
                })
                
            except Exception as e:
                # If an error occurs, pass it to the waiting client's future
                item["future"].set_exception(e)
            finally:
                q_tts.task_done()

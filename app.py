"""
SmartHealth Community AI — Layer 6: Master Flask Ingress Server & Worker Bootstrapper
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: app.py
Responsibility: Runs the 4 SEDA worker loops in a background thread,
handles audio/text ingress, and returns DUAL OUTPUTS (Spoken Voice Note + Structured Clinical Text).
"""

import os
import io
import time
import asyncio
import threading
import concurrent.futures
import torch
import torchaudio
from flask import Flask, request, jsonify, send_from_directory

# Import our modular pipeline engine and worker loops
from core_engine import q_stt, q_rag, create_patient_packet
from stt_model import stt_model_worker
from rag_model import rag_model_worker
from llm_gateway import llm_gateway_worker
from tts_model import tts_model_worker, OUTPUT_AUDIO_DIR

# =============================================================================
# 1. FLASK APP INITIALIZATION
# =============================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max audio upload limit

# Dedicated asyncio event loop for SEDA worker stations
pipeline_loop = asyncio.new_event_loop()


# =============================================================================
# 2. BOOTSTRAP ALL 4 MODEL WORKERS PERMANENTLY IN BACKGROUND THREAD
# =============================================================================
def start_background_workers(loop: asyncio.AbstractEventLoop):
    """
    Runs the 4 SEDA worker loops inside a dedicated background thread.
    These workers run 24/7 in memory and never terminate between requests.
    """
    asyncio.set_event_loop(loop)
    print("\n" + "="*70)
    print("🚀 [SmartHealth Boot] Spawning 4 Staged Event-Driven Workers in Flask...")
    
    loop.create_task(stt_model_worker(max_batch_size=4, timeout_ms=50))
    loop.create_task(rag_model_worker(max_batch_size=16, timeout_ms=15))
    loop.create_task(llm_gateway_worker(max_batch_size=4, timeout_ms=30))
    loop.create_task(tts_model_worker(max_batch_size=4, timeout_ms=40))
    
    print("✓ Station 1 (STT / LID)      : ACTIVE (q_stt)")
    print("✓ Station 2 (RAG / Safety)   : ACTIVE (q_rag)")
    print("✓ Station 3 (LLM Gateway)    : ACTIVE (q_llm)")
    print("✓ Station 4 (TTS Voice)      : ACTIVE (q_tts)")
    print("✓ All workers active and listening to RAM queues.")
    print("="*70 + "\n")
    
    loop.run_forever()

# Launch background workers on application startup
worker_thread = threading.Thread(target=start_background_workers, args=(pipeline_loop,), daemon=True)
worker_thread.start()


# =============================================================================
# 3. HELPER: AUDIO NORMALIZATION TO 16,000 HZ
# =============================================================================
def resample_inbound_audio(audio_bytes: bytes) -> tuple:
    """
    Converts uploaded voice notes (.wav, .ogg, .mp3) into mono 16,000 Hz NumPy array.
    """
    buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buffer)
    
    # Convert multi-channel stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample to strictly 16,000 Hz (required by Meta MMS)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        
    return waveform.squeeze().numpy(), 16000


# =============================================================================
# 4. FLASK API ENDPOINTS (DUAL TEXT + AUDIO DELIVERY)
# =============================================================================

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint for Render monitoring."""
    return jsonify({
        "status": "ONLINE",
        "service": "SmartHealth Community AI Engine (Flask)",
        "supported_dialects": ["pcm (Nigerian Pidgin)", "hau (Hausa)", "eng (English)"],
        "pipeline_architecture": "Staged Event-Driven Architecture (SEDA / Pipes & Filters)"
    })


@app.route("/api/triage/audio", methods=["POST"])
def triage_audio_endpoint():
    """
    Primary Voice Note Ingress (Mobile App / Telegram Webhook).
    - Ingests raw voice note
    - Processes through: LID -> STT -> Safety Gate -> RAG -> LLM -> TTS
    - Returns BOTH clinical text advice and the synthesized audio voice note URL!
    """
    audio_file = request.files.get("file") or request.files.get("audio")
    if not audio_file:
        return jsonify({"status": "ERROR", "error": "Audio file is required (form key: 'file' or 'audio')."}), 400
        
    raw_bytes = audio_file.read()
    if len(raw_bytes) < 1000:
        return jsonify({"status": "ERROR", "error": "Audio file is too short or corrupted."}), 400
        
    try:
        # Step 1: Normalize audio to 16kHz
        audio_16k_array, _ = resample_inbound_audio(raw_bytes)
        
        # Step 2: Create thread-safe concurrent.futures.Future
        future = concurrent.futures.Future()
        packet = create_patient_packet(audio_16k_array, future)
        
        # Step 3: DROP INTO QUEUE 1 (STT Station) in the background loop
        asyncio.run_coroutine_threadsafe(q_stt.put(packet), pipeline_loop)
        
        # Step 4: AWAIT RESOLUTION FROM STATION 4 (TTS Station)
        result = future.result(timeout=60.0)
        
        # Format audio download URL
        audio_filename = os.path.basename(result["audio_file_path"])
        audio_url = f"/api/audio/{audio_filename}"
        
        # Step 5: RETURN DUAL PAYLOAD (TEXT + SPEECH)
        return jsonify({
            "status": "COMPLETED_SUCCESS",
            "trace_id": result["trace_id"],
            "detected_dialect": result["dialect"],
            "patient_transcription": result["transcription"],
            "clinical_safety": {
                "is_emergency": result["is_emergency"],
                "emergency_trigger": result.get("emergency_trigger"),
                "fmoh_guideline_matched": result["matched_guideline"]
            },
            "delivery_payload": {
                "clinical_advice_text": result["clinical_advice"],
                "audio_voice_note_url": audio_url,
                "audio_duration_seconds": result.get("audio_duration", 0)
            },
            "telemetry": {
                "total_pipeline_latency_ms": result["total_pipeline_latency_ms"]
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "ERROR", "error": f"Pipeline execution error: {str(e)}"}), 500


@app.route("/api/triage/text", methods=["POST"])
def triage_text_endpoint():
    """
    Direct Text Ingress (Bypasses STT for sub-second text triage).
    Drops directly into Station 2 (RAG Queue) and returns Text + Audio.
    """
    query_text = request.form.get("query_text") or (request.json.get("query_text") if request.is_json else None)
    dialect = request.form.get("dialect") or (request.json.get("dialect") if request.is_json else "pcm")
    
    if not query_text or not query_text.strip():
        return jsonify({"status": "ERROR", "error": "Field 'query_text' cannot be empty."}), 400
        
    try:
        # Create thread-safe concurrent.futures.Future
        future = concurrent.futures.Future()
        packet = {
            "trace_id": f"trace_{time.time_ns()}",
            "timestamp_start": time.perf_counter(),
            "audio_16k": None,
            "detected_dialect": dialect,
            "transcription": query_text.strip(),
            "future": future
        }
        
        # DROP DIRECTLY INTO QUEUE 2 (RAG Station)
        asyncio.run_coroutine_threadsafe(q_rag.put(packet), pipeline_loop)
        
        # Await completion from Station 4 (TTS)
        result = future.result(timeout=60.0)
        audio_filename = os.path.basename(result["audio_file_path"])
        audio_url = f"/api/audio/{audio_filename}"
        
        return jsonify({
            "status": "COMPLETED_SUCCESS",
            "trace_id": result["trace_id"],
            "detected_dialect": dialect,
            "patient_transcription": result["transcription"],
            "clinical_safety": {
                "is_emergency": result["is_emergency"],
                "emergency_trigger": result.get("emergency_trigger"),
                "fmoh_guideline_matched": result["matched_guideline"]
            },
            "delivery_payload": {
                "clinical_advice_text": result["clinical_advice"],
                "audio_voice_note_url": audio_url,
                "audio_duration_seconds": result.get("audio_duration", 0)
            },
            "telemetry": {
                "total_pipeline_latency_ms": result["total_pipeline_latency_ms"]
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "ERROR", "error": f"Pipeline execution error: {str(e)}"}), 500


@app.route("/api/audio/<filename>", methods=["GET"])
def get_audio_file(filename: str):
    """Serves the generated .wav/.ogg voice notes to mobile/web clients."""
    file_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Audio file not found."}), 404
    return send_from_directory(OUTPUT_AUDIO_DIR, filename, mimetype="audio/wav")


# =============================================================================
# 5. LOCAL RUNNER
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[SmartHealth Flask] Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

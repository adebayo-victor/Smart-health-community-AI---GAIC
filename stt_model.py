"""
SmartHealth Community AI — Layer 2: Speech Models & Station 1 (STT / LID)
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: smarthealth-community-ai/models.py
Responsibility: Loads speech recognition (STT), language identification (LID), 
and text-to-speech (TTS) models. Implements the Station 1 worker loop.
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

# =============================================================================
# 0. HARDWARE DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SmartHealth Models] Initializing models on hardware: {DEVICE.upper()}")


# =============================================================================
# 1. MODEL LOADER: LANGUAGE IDENTIFICATION (Audio LID)
# =============================================================================
print("[1/2 Loading] Meta MMS-LID (`facebook/mms-lid-126`)...")
lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-126")
lid_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-126").to(DEVICE)
lid_model.eval()


# =============================================================================
# 2. MODEL LOADER: SPEECH-TO-TEXT (Meta MMS-1B Multilingual STT)
# =============================================================================
print("[2/2 Loading] Meta MMS-1B STT (`facebook/mms-1b-all`)...")
stt_processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(DEVICE)
stt_model.eval()


# =============================================================================
# 3. HELPER FUNCTIONS FOR INFERENCE
# =============================================================================
def detect_language(audio_16k_array) -> str:
    """
    Analyzes the first 2.5 to 3.0 seconds (48,000 samples) of audio
    to identify if the dialect is Hausa ('hau') or Nigerian Pidgin ('pcm').
    """
    sample_window = audio_16k_array[:48000]
    inputs = lid_processor(sample_window, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        logits = lid_model(**inputs).logits
        
    probabilities = torch.softmax(logits, dim=-1)
    predicted_id = torch.argmax(probabilities, dim=-1).item()
    detected_tag = lid_model.config.id2label[predicted_id].lower()
    
    # Map to standard project dialect codes
    if "hau" in detected_tag:
        return "hau"
    elif "pcm" in detected_tag or "eng" in detected_tag:
        return "pcm"
    
    # Default fallback to Pidgin for sub-Saharan regional dialect ambiguity
    return "pcm"


def transcribe_mms(audio_16k_array, dialect_code: str) -> str:
    """
    Transcribes a 16kHz audio array using Meta MMS-1B by hot-swapping
    the target language adapter layer ('hau' vs 'pcm').
    """
    # 1. Hot-swap tokenizer and acoustic adapter for target language
    stt_processor.tokenizer.set_target_lang(dialect_code)
    stt_model.load_adapter(dialect_code)
    
    # 2. Feature extraction and tensor forward pass
    inputs = stt_processor(audio_16k_array, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        logits = stt_model(inputs.input_values).logits
        
    # 3. Decode CTC predicted tokens into verbatim text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = stt_processor.decode(predicted_ids[0]).strip()
    
    return transcription


# =============================================================================
# 4. STATION 1 ASYNC WORKER LOOP (The Ear)
# =============================================================================
async def stt_model_worker(max_batch_size: int = 4, timeout_ms: int = 50):
    """
    Station 1 Background Worker.
    - Listens to: `q_stt`
    - Trigger Condition: N=4 audio items OR T=50ms timeout
    - Action: Detects language, transcribes audio, and deposits to `q_rag`
    """
    print(f"[Worker 1: STT/LID Active] Ready on {DEVICE.upper()} (N={max_batch_size}, T={timeout_ms}ms)")
    
    while True:
        # Collect batch using our N or T rule
        batch = await collect_batch(q_stt, max_batch_size, timeout_ms)
        
        for item in batch:
            try:
                raw_audio = item["audio_16k"]
                
                # Step 1: Detect Dialect (LID)
                detected_dialect = detect_language(raw_audio)
                
                # Step 2: Transcribe (MMS-1B STT)
                transcription_text = transcribe_mms(raw_audio, detected_dialect)
                
                # Step 3: Populate packet
                item["detected_dialect"] = detected_dialect
                item["transcription"] = transcription_text
                
                # Step 4: HAND OFF TO STATION 2 (RAG Queue)
                await q_rag.put(item)
                
            except Exception as e:
                # If an error occurs, pass it to the waiting client's future
                item["future"].set_exception(e)
            finally:
                # Signal task completion for this queue item
                q_stt.task_done()

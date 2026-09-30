"""
SmartHealth Community AI — Layer 1: Core Queue Engine
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: smarthealth-community-ai/core_engine.py
Responsibility: Defines the in-memory SEDA queues, the data packet schema,
and the dual-condition batch collector (N items or T ms timeout).
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any

# =============================================================================
# 1. INTER-STAGE IN-MEMORY QUEUES (The Data Pipes)
# =============================================================================
# These queues reside entirely in RAM. No temporary files are written to disk.

# Queue for Station 1: Receives raw audio arrays from user ingress -> feeds STT/LID
q_stt = asyncio.Queue()

# Queue for Station 2: Receives transcribed text -> feeds Safety Gate & RAG
q_rag = asyncio.Queue()

# Queue for Station 3: Receives [Query + STG Guideline] -> feeds LiteLLM Gateway
q_llm = asyncio.Queue()

# Queue for Station 4: Receives verified clinical text -> feeds Meta MMS-TTS
q_tts = asyncio.Queue()


# =============================================================================
# 2. STANDARDIZED DATA PACKET (The Relay Baton)
# =============================================================================
def create_patient_packet(audio_16k_array, future: asyncio.Future) -> Dict[str, Any]:
    """
    Creates the standardized dictionary that travels through RAM across stations.
    
    Parameters:
    - audio_16k_array: NumPy array or PyTorch tensor sampled at 16,000 Hz.
    - future: An asyncio.Future promise resolved when Station 4 (TTS) completes.
    
    Returns:
    - Standardized data packet dictionary.
    """
    return {
        # Unique trace identifier for auditing and telemetry (NDPR compliance)
        "trace_id": f"trace_{uuid.uuid4().hex[:8]}",
        
        # High-resolution timestamp when the request entered the server
        "timestamp_start": time.perf_counter(),
        
        # Raw Inbound Data
        "audio_16k": audio_16k_array,
        
        # Fields populated sequentially by each downstream model worker:
        "detected_dialect": None,    # Set by Station 1 (e.g., 'pcm' or 'hau')
        "transcription": None,       # Set by Station 1 (Verbatim text)
        "is_emergency": False,       # Set by Station 2 (True if red flag detected)
        "matched_guideline": None,   # Set by Station 2 (STG chunk ID & protocol)
        "clinical_advice": None,     # Set by Station 3 (Text advice from LLM)
        "outbound_audio_path": None, # Set by Station 4 (Path to generated .wav/.ogg)
        
        # The Promise object returned to the waiting Telegram/Web client
        "future": future
    }


# =============================================================================
# 3. DUAL-CONDITION BATCH COLLECTOR (The N or T Trigger Rule)
# =============================================================================
async def collect_batch(queue: asyncio.Queue, max_batch_size: int, timeout_ms: int) -> List[Dict[str, Any]]:
    """
    Collects items from an in-memory queue until either:
    - Condition A (Capacity): Queue collects `max_batch_size` items (N items).
    - Condition B (Timeout): Timer reaches `timeout_ms` (T milliseconds).
    
    This guarantees maximum GPU batch throughput during peak clinic hours,
    while completely eliminating the '2:00 AM lone user starvation problem'.
    
    Parameters:
    - queue: The asyncio.Queue to draw items from.
    - max_batch_size: Maximum batch size threshold (N).
    - timeout_ms: Maximum wait timeout in milliseconds (T).
    
    Returns:
    - A list of data packets ready for batch inference.
    """
    timeout_seconds = timeout_ms / 1000.0
    batch = []

    # STEP 1: Block asynchronously until at least the first item arrives.
    # When no users are online, the worker pauses here with 0% CPU consumption.
    first_item = await queue.get()
    batch.append(first_item)
    start_time = time.perf_counter()

    # STEP 2: Collect additional items until batch is full OR timeout expires.
    while len(batch) < max_batch_size:
        elapsed = time.perf_counter() - start_time
        remaining_time = timeout_seconds - elapsed
        
        # If timeout has elapsed, fire the batch immediately!
        if remaining_time <= 0:
            break
            
        try:
            # Wait for next item, capped by the remaining timeout budget
            next_item = await asyncio.wait_for(queue.get(), timeout=remaining_time)
            batch.append(next_item)
        except asyncio.TimeoutError:
            # Timeout reached before queue filled to max_batch_size -> Fire batch!
            break

    return batch

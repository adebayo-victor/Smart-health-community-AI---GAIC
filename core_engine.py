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

q_stt = asyncio.Queue()
q_rag = asyncio.Queue()
q_llm = asyncio.Queue()
q_tts = asyncio.Queue()

# =============================================================================
# 2. STANDARDIZED DATA PACKET (The Relay Baton)
# =============================================================================
def create_patient_packet(audio_16k_array, future: asyncio.Future) -> Dict[str, Any]:
    """
    Creates the standardized dictionary that travels through RAM across stations.
    """
    return {
        "trace_id": f"trace_{uuid.uuid4().hex[:8]}",
        "timestamp_start": time.perf_counter(),
        "audio_16k": audio_16k_array,
        
        # Fields populated sequentially by each downstream model worker:
        "detected_dialect": None,    
        "transcription": None,       
        "is_emergency": False,       
        "matched_guidelines": None,  # FIXED: Pluralized to match rag_model.py output
        "clinical_advice": None,     
        "outbound_audio_path": None, 
        
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
    """
    timeout_seconds = timeout_ms / 1000.0
    batch = []

    # STEP 1: Block asynchronously until at least the first item arrives.
    first_item = await queue.get()
    batch.append(first_item)
    start_time = time.perf_counter()

    # STEP 2: Collect additional items until batch is full OR timeout expires.
    while len(batch) < max_batch_size:
        elapsed = time.perf_counter() - start_time
        remaining_time = timeout_seconds - elapsed
        
        if remaining_time <= 0:
            break
            
        try:
            next_item = await asyncio.wait_for(queue.get(), timeout=remaining_time)
            batch.append(next_item)
        except asyncio.TimeoutError:
            break

    return batch

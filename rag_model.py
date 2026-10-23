"""
SmartHealth Community AI — Layer 3: PDF Knowledge Base RAG & Station 2 Worker
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: smarthealth-community-ai/rag_model.py
Responsibility: 
1. Ingests Nigerian Standard Treatment Guidelines from data/stg_guidelines/*.pdf.
2. Auto-builds and checks freshness of the PyTorch embeddings cache.
3. Runs Station 2 Background Worker: Translates to English -> Safety Gate -> Top-3 Vector Retrieval -> deposits into `q_llm`.
"""

import os
import glob
import asyncio
import re
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline # For zero-shot emergency detection
from core_engine import q_rag, q_llm, collect_batch
from translator_model import translate_to_english

# =============================================================================
# 0. HARDWARE & DIRECTORY CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "data/stg_guidelines")
CACHE_FILE_PATH = os.path.join(PDF_DIRECTORY, "stg_embeddings_cache.pt")

print(f"[SmartHealth RAG] Initializing retrieval engine on: {DEVICE.upper()}")
os.makedirs(PDF_DIRECTORY, exist_ok=True)

# =============================================================================
# 1. LOAD EMBEDDING MODEL & ZERO-SHOT CLASSIFIER
# =============================================================================
print("[1/3 Loading] all-MiniLM-L6-v2 Embedding Model...")
rag_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

print("[2/3 Loading] Zero-Shot Emergency Classifier (No fine-tuning needed)...")
# This model can classify text into ANY labels we give it, in English.
# We use a small, fast version to keep latency under 50ms.
emergency_classifier = pipeline(
    "zero-shot-classification", 
    model="typeform/distilbert-base-uncased-mnli", 
    device=0 if DEVICE == "cuda" else -1
)

EMERGENCY_LABELS = ["urgent medical emergency", "routine health inquiry", "general conversation"]

# =============================================================================
# 2. PDF PARSER & SLIDING-WINDOW CHUNKER (Kept as requested)
# =============================================================================
def extract_and_chunk_pdfs(pdf_folder_path: str, chunk_size: int = 350, chunk_overlap: int = 50) -> list[dict]:
    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(
            f"\n[CRITICAL ERROR] No PDF guidelines found in '{pdf_folder_path}'.\n"
            f"Please place your official Nigerian Standard Treatment Guideline PDF files inside this folder."
        )
    
    chunks = []
    print(f"[SmartHealth RAG] Found {len(pdf_files)} PDF file(s). Extracting text...")
    
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        print(f" -> Parsing PDF: {pdf_name}...")
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if not page_text or len(page_text.strip()) < 20:
                    continue
                
                # Your original word-based sliding window
                words = page_text.split()
                for i in range(0, len(words), chunk_size - chunk_overlap):
                    chunk_words = words[i : i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    if len(chunk_text.strip()) > 30:
                        chunks.append({
                            "id": f"{pdf_name}_p{page_num}_c{len(chunks)+1}",
                            "source": pdf_name,
                            "page": page_num,
                            "text": chunk_text
                        })
        except Exception as e:
            print(f"[Error parsing {pdf_name}]: {e}")
            
    print(f"[SmartHealth RAG] Extracted and constructed {len(chunks)} total guideline chunks.")
    return chunks

# =============================================================================
# 3. SELF-HEALING / AUTO-INVALIDATING .PT CACHE LOADER
# =============================================================================
def init_or_load_knowledge_base(pdf_dir: str, cache_path: str, embedding_model, device: str):
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not os.path.exists(cache_path) and not pdf_files:
        raise FileNotFoundError(
            f"\n[CRITICAL ERROR] No PDF guidelines found in '{pdf_dir}' and no existing cache.\n"
            f"Please add STG PDF files to '{pdf_dir}'."
        )

    if os.path.exists(cache_path) and pdf_files:
        cache_mtime = os.path.getmtime(cache_path)
        latest_pdf_mtime = max(os.path.getmtime(pdf) for pdf in pdf_files)
        
        if latest_pdf_mtime <= cache_mtime:
            print(f"[3/3 Loading Cache] Found fresh pre-computed cache. Loading into RAM...")
            cache_data = torch.load(cache_path, map_location=device, weights_only=False)
            return cache_data["chunks"], cache_data["embeddings"].to(device)
        else:
            print("[3/3 Cache Invalidated] New PDF files detected! Rebuilding knowledge base...")

    print(f"[3/3 Cache Miss] Building knowledge base from raw PDFs in '{pdf_dir}'...")
    chunks_db = extract_and_chunk_pdfs(pdf_dir)
    
    print("[4/4 Vectorizing] Computing dense embeddings for all chunks...")
    corpus_texts = [c["text"] for c in chunks_db]
    embeddings = embedding_model.encode(corpus_texts, convert_to_tensor=True, device=device)
    
    print(f"[SmartHealth RAG] Saving computed vectors to cache: '{cache_path}'...")
    torch.save({"chunks": chunks_db, "embeddings": embeddings.cpu()}, cache_path)
    
    return chunks_db, embeddings.to(device)

# Initialize Knowledge Base on module import
STG_CHUNKS_DATABASE, stg_embeddings = init_or_load_knowledge_base(
    pdf_dir=PDF_DIRECTORY,
    cache_path=CACHE_FILE_PATH,
    embedding_model=rag_model,
    device=DEVICE
)

# =============================================================================
# 4. HELPER FUNCTIONS: SAFETY GATE & TOP-K VECTOR SEARCH
# =============================================================================
def check_emergency(text: str) -> tuple[bool, str]:
    """
    Two-Layer Safety Gate:
    Layer 1: Instant Regex with word boundaries (\b) to prevent partial word false positives.
    Layer 2: Zero-Shot Classification on the text to catch contextual emergencies.
    """
    text_lower = text.lower()
    
    # Layer 1: Deterministic Regex (Sub-1ms)
    emergency_keywords = [
        r"\bzubar jini\b", r"\bjini da yawa\b", r"\bbleeding heavily\b", 
        r"\bblood clot\b", r"\bconvulsion\b", r"\bfaint\b", r"\bgreen water\b"
    ]
    for pattern in emergency_keywords:
        if re.search(pattern, text_lower):
            return True, f"Regex Match: {pattern}"
            
    # Layer 2: Zero-Shot Classifier (Catches context like "my water broke and it's green")
    # We run this on the English translation for maximum accuracy
    try:
        result = emergency_classifier(text, candidate_labels=EMERGENCY_LABELS)
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        # If the model is > 85% confident it's an emergency, trigger it
        if top_label == "urgent medical emergency" and top_score > 0.85:
            return True, f"Zero-Shot Match: {top_label} ({top_score:.2f})"
    except Exception as e:
        print(f"[RAG Warning] Zero-shot classifier failed: {e}")
        
    return False, ""

def retrieve_top_k_stg_guidelines(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Embeds query and computes Cosine Similarity, returning Top-K chunks 
    to serve as robust "memory" for the LLM.
    """
    query_vector = rag_model.encode(query_text, convert_to_tensor=True, device=DEVICE)
    cosine_scores = util.cos_sim(query_vector, stg_embeddings)[0]
    
    # Get indices of the top K highest scores
    top_k_indices = torch.topk(cosine_scores, k=min(top_k, len(STG_CHUNKS_DATABASE))).indices
    
    return [STG_CHUNKS_DATABASE[int(idx)] for idx in top_k_indices]

# =============================================================================
# 5. STATION 2 ASYNC WORKER LOOP (The Library & Safety Gate)
# =============================================================================
async def rag_model_worker(max_batch_size: int = 16, timeout_ms: int = 15):
    print(f"[Worker 2: PDF RAG/Safety Active] Ready on {DEVICE.upper()} (N={max_batch_size}, T={timeout_ms}ms)")
    
    while True:
        batch = await collect_batch(q_rag, max_batch_size, timeout_ms)
        
        # CRITICAL FIX 1: Translate all queries to English FIRST for accurate RAG retrieval
        english_queries = [
            translate_to_english(item["transcription"], item["detected_dialect"]) 
            for item in batch
        ]
        
        # Batch vectorization of the ENGLISH queries
        query_vectors = rag_model.encode(english_queries, convert_to_tensor=True, device=DEVICE)
        
        for idx, item in enumerate(batch):
            try:
                transcript = item["transcription"]
                english_query = english_queries[idx]
                
                # Step 1: Run Two-Layer Safety Filter on the English query
                is_emergency, trigger_reason = check_emergency(english_query)
                
                # Step 2: Retrieve Top-3 Matching Chunks from Vector Matrix (LLM Memory)
                top_k_chunks = retrieve_top_k_stg_guidelines(english_query, top_k=3)
                
                # Step 3: Populate packet
                item["is_emergency"] = is_emergency
                item["emergency_trigger"] = trigger_reason
                item["matched_guidelines"] = top_k_chunks # Changed to plural for Top-K
                
                # Step 4: HAND OFF TO STATION 3 (LLM Gateway Queue)
                await q_llm.put(item)
                
            except Exception as e:
                item["future"].set_exception(e)
            finally:
                q_rag.task_done()

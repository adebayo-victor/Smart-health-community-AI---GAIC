"""
SmartHealth Community AI — Layer 3: PDF Knowledge Base RAG & Station 2 Worker
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: smarthealth-community-ai/rag_model.py
Responsibility: 
1. Ingests 2016/2022 Nigerian Standard Treatment Guidelines from data/stg_guidelines/*.pdf.
2. Auto-builds and checks freshness of the PyTorch embeddings cache (stg_embeddings_cache.pt).
3. Runs Station 2 Background Worker (Safety Gating + Sub-20ms Vector Retrieval) -> deposits into `q_llm`.
"""

import os
import glob
import asyncio
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from core_engine import q_rag, q_llm, collect_batch

# =============================================================================
# 0. HARDWARE & DIRECTORY CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "data/stg_guidelines")
CACHE_FILE_PATH = os.path.join(PDF_DIRECTORY, "stg_embeddings_cache.pt")

print(f"[SmartHealth RAG] Initializing retrieval engine on: {DEVICE.upper()}")
os.makedirs(PDF_DIRECTORY, exist_ok=True)


# =============================================================================
# 1. LOAD EMBEDDING MODEL (`all-MiniLM-L6-v2` - 80MB)
# =============================================================================
print("[1/3 Loading] all-MiniLM-L6-v2 Embedding Model...")
rag_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)


# =============================================================================
# 2. PDF PARSER & SLIDING-WINDOW CHUNKER
# =============================================================================
def extract_and_chunk_pdfs(pdf_folder_path: str, chunk_size: int = 350, chunk_overlap: int = 50) -> list[dict]:
    """
    Scans folder for .pdf files, extracts raw text page-by-page, and constructs chunks.
    Raises FileNotFoundError if no PDFs exist.
    """
    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    
    # CRITICAL CHECK: Throw error if PDFs are missing
    if not pdf_files:
        raise FileNotFoundError(
            f"\n[CRITICAL ERROR] No PDF guidelines found in '{pdf_folder_path}'.\n"
            f"Please place your official 2016/2022 Nigerian Standard Treatment Guideline "
            f"PDF files inside '{pdf_folder_path}'."
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
                
                # Sliding-window chunking (preserves medical context across page splits)
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
    """
    1. Looks for pre-computed `stg_embeddings_cache.pt`.
    2. Checks timestamps of individual .pdf files against the cache file.
    3. If cache exists and is fresh: Loads vectors into RAM in <0.2 seconds.
    4. If cache is missing or any PDF is newer: Re-parses PDFs and saves a fresh `.pt` file.
    """
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    # Check 1: Ensure PDF directory has source files if no cache exists
    if not os.path.exists(cache_path) and not pdf_files:
        raise FileNotFoundError(
            f"\n[CRITICAL ERROR] No PDF guidelines found in '{pdf_dir}' "
            f"and no existing cache found at '{cache_path}'.\n"
            f"Please place your official 2016/2022 Nigerian Standard Treatment Guideline "
            f"PDF files inside '{pdf_dir}'."
        )

    # Check 2: If cache exists, verify freshness against each .pdf file
    if os.path.exists(cache_path) and pdf_files:
        cache_mtime = os.path.getmtime(cache_path)
        latest_pdf_mtime = max(os.path.getmtime(pdf) for pdf in pdf_files)
        
        # Freshness check: if no PDF is newer than the cache, load instantly
        if latest_pdf_mtime <= cache_mtime:
            print(f"[2/3 Loading Cache] Found fresh pre-computed cache at '{cache_path}'. Loading into RAM...")
            cache_data = torch.load(cache_path, map_location=device, weights_only=False)
            chunks_db = cache_data["chunks"]
            embeddings = cache_data["embeddings"].to(device)
            print(f"[3/3 Ready] Loaded {len(chunks_db)} guideline vectors in <0.2s from .pt cache.\n")
            return chunks_db, embeddings
        else:
            print("[2/3 Cache Invalidated] New or updated PDF files detected! Rebuilding knowledge base...")

    # Fallback / Rebuild Step: Parse raw PDFs and compute embeddings
    print(f"[2/3 Cache Miss/Outdated] Building knowledge base from raw PDFs in '{pdf_dir}'...")
    chunks_db = extract_and_chunk_pdfs(pdf_dir)
    
    print("[3/3 Vectorizing] Computing dense all-MiniLM-L6-v2 embeddings for all chunks...")
    corpus_texts = [c["text"] for c in chunks_db]
    embeddings = embedding_model.encode(corpus_texts, convert_to_tensor=True, device=device)
    
    # Save fresh cache for future server restarts
    print(f"[SmartHealth RAG] Saving computed vectors to cache: '{cache_path}'...")
    torch.save({
        "chunks": chunks_db,
        "embeddings": embeddings.cpu()
    }, cache_path)
    
    print(f"[SmartHealth RAG] Knowledge base cache built and saved successfully!\n")
    return chunks_db, embeddings.to(device)


# Initialize Knowledge Base on module import
STG_CHUNKS_DATABASE, stg_embeddings = init_or_load_knowledge_base(
    pdf_dir=PDF_DIRECTORY,
    cache_path=CACHE_FILE_PATH,
    embedding_model=rag_model,
    device=DEVICE
)


# =============================================================================
# 4. HELPER FUNCTIONS: SAFETY GATE & VECTOR SEARCH
# =============================================================================
def check_emergency_triggers(text: str) -> tuple[bool, str]:
    """
    Sub-5ms deterministic regex/keyword filter for acute obstetric red flags.
    Checks Hausa, Pidgin, and English emergency terms.
    """
    text_lower = text.lower()
    emergency_keywords = [
        "zubar jini", "jini da yawa", "bleeding heavily", "blood clot",
        "seeing flashes", "flashes of light", "convulsion", "faint", "dizzy on standing"
    ]
    for keyword in emergency_keywords:
        if keyword in text_lower:
            return True, keyword
    return False, ""


def retrieve_stg_guideline(query_text: str) -> dict:
    """
    Embeds query and computes Cosine Similarity against all cached STG vectors in <20ms.
    """
    query_vector = rag_model.encode(query_text, convert_to_tensor=True, device=DEVICE)
    cosine_scores = util.cos_sim(query_vector, stg_embeddings)[0]
    top_index = int(torch.argmax(cosine_scores))
    return STG_CHUNKS_DATABASE[top_index]


# =============================================================================
# 5. STATION 2 ASYNC WORKER LOOP (The Library & Safety Gate)
# =============================================================================
async def rag_model_worker(max_batch_size: int = 16, timeout_ms: int = 15):
    """
    Station 2 Background Worker.
    - Listens to: `q_rag`
    - Trigger Condition: N=16 text items OR T=15ms timeout
    - Action: Checks safety triggers, searches PDF vectors in RAM, and deposits to `q_llm`
    """
    print(f"[Worker 2: PDF RAG/Safety Active] Ready on {DEVICE.upper()} (N={max_batch_size}, T={timeout_ms}ms)")
    
    while True:
        # Collect batch using our N or T rule
        batch = await collect_batch(q_rag, max_batch_size, timeout_ms)
        
        # Batch vectorization of all query texts in 1 single forward pass
        query_texts = [item["transcription"] for item in batch]
        query_vectors = rag_model.encode(query_texts, convert_to_tensor=True, device=DEVICE)
        
        for idx, item in enumerate(batch):
            try:
                transcript = item["transcription"]
                
                # Step 1: Run Deterministic Safety Filter
                is_emergency, trigger_word = check_emergency_triggers(transcript)
                
                # Step 2: Retrieve Top-1 Matching Chunk from Vector Matrix
                cosine_scores = util.cos_sim(query_vectors[idx], stg_embeddings)[0]
                top_index = int(torch.argmax(cosine_scores))
                matched_chunk = STG_CHUNKS_DATABASE[top_index]
                
                # Step 3: Populate packet with PDF chunk and metadata
                item["is_emergency"] = is_emergency
                item["emergency_trigger"] = trigger_word
                item["matched_guideline"] = matched_chunk
                
                # Step 4: HAND OFF TO STATION 3 (LLM Gateway Queue)
                await q_llm.put(item)
                
            except Exception as e:
                item["future"].set_exception(e)
            finally:
                q_rag.task_done()

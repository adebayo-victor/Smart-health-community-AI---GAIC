"""
SmartHealth Community AI — Embeddings Cache Generator
Runs ONCE offline to parse PDFs and create stg_embeddings_cache.pt
Then the main app.py just loads it with no disk/memory issues.
"""

import os
import sys
import glob
import torch
import gc
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Ensure we can import from the root directory when running from utils/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import MODELS, PATHS, PARAMS

# =============================================================================
# CONFIG (Now Centralized)
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PDF_DIRECTORY = PATHS["pdf_directory"]
CACHE_FILE_PATH = PATHS["cache_file_path"]

print(f"[Embeddings Generator] Running on: {DEVICE.upper()}")
os.makedirs(PDF_DIRECTORY, exist_ok=True)


# =============================================================================
# 1. LOAD EMBEDDING MODEL
# =============================================================================
print(f"[1/3 Loading] Embedding Model (`{MODELS['embedding']}`)...")
rag_model = SentenceTransformer(MODELS["embedding"], device=DEVICE)


# =============================================================================
# 2. PDF PARSER & SLIDING-WINDOW CHUNKER (Memory-Safe with GC)
# =============================================================================
def extract_and_chunk_pdfs(pdf_folder_path: str) -> list[dict]:
    """
    Scans folder for .pdf files, extracts raw text page-by-page with explicit garbage collection.
    Processes one page at a time to avoid memory bloat.
    """
    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"\n[ERROR] No PDF guidelines found in '{pdf_folder_path}'.\n"
            f"Please place your official 2016/2022 Nigerian Standard Treatment Guideline "
            f"PDF files inside '{pdf_folder_path}'."
        )

    chunks = []
    print(f"[2/3 Parsing] Found {len(pdf_files)} PDF file(s). Extracting text...\n")

    # Pull chunking params from config
    chunk_size = PARAMS["chunk_size"]
    chunk_overlap = PARAMS["chunk_overlap"]

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        print(f" -> Parsing PDF: {pdf_name}...")

        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            print(f"    Total pages: {total_pages}")

            # Process pages one at a time to avoid memory explosion
            for page_num in range(total_pages):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()

                    if not page_text or len(page_text.strip()) < 20:
                        continue

                    # Sliding-window chunking
                    words = page_text.split()
                    for i in range(0, len(words), chunk_size - chunk_overlap):
                        chunk_words = words[i : i + chunk_size]
                        chunk_text = " ".join(chunk_words)

                        if len(chunk_text.strip()) > 30:
                            chunks.append({
                                "id": f"{pdf_name}_p{page_num+1}_c{len(chunks)+1}",
                                "source": pdf_name,
                                "page": page_num + 1,
                                "text": chunk_text
                            })

                    # Explicit cleanup every 10 pages
                    if (page_num + 1) % 10 == 0:
                        del page
                        gc.collect()
                        print(f"    ✓ Processed {page_num + 1}/{total_pages} pages...")

                except Exception as page_err:
                    print(f"    [Warning] Skipping page {page_num + 1}: {page_err}")
                    continue

            print(f"    ✓ Extracted {sum(1 for c in chunks if c['source'] == pdf_name)} chunks from {pdf_name}\n")

        except Exception as e:
            print(f"[Error parsing {pdf_name}]: {e}\n")
            continue

    print(f"[Embeddings Generator] Total extracted: {len(chunks)} guideline chunks.\n")
    return chunks


# =============================================================================
# 3. GENERATE EMBEDDINGS & SAVE CACHE
# =============================================================================
def generate_and_save_cache(pdf_dir: str, cache_path: str, embedding_model, device: str):
    """
    Parses PDFs, computes embeddings in batches, saves to .pt file.
    """
    print("[2/3 Extracting] Parsing PDFs and chunking text...")
    chunks_db = extract_and_chunk_pdfs(pdf_dir)

    print("[3/3 Vectorizing] Computing embeddings (this may take a few minutes)...")
    corpus_texts = [c["text"] for c in chunks_db]

    # Encode in batches to avoid memory overload
    batch_size = 64
    all_embeddings = []

    for i in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[i : i + batch_size]
        print(f"  Encoding batch {i // batch_size + 1}/{(len(corpus_texts) + batch_size - 1) // batch_size}...")
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=True, device=device)
        all_embeddings.append(batch_embeddings)
        gc.collect()

    embeddings = torch.cat(all_embeddings, dim=0)

    # Save cache
    print(f"\n[Generator] Saving embeddings cache to '{cache_path}'...")
    torch.save({
        "chunks": chunks_db,
        "embeddings": embeddings.cpu()
    }, cache_path)

    print(f"✓ Cache saved successfully! ({len(chunks_db)} chunks, {embeddings.shape} embeddings)\n")
    print(f"You can now run: python app.py")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SmartHealth Embeddings Cache Generator")
    print("="*70 + "\n")

    generate_and_save_cache(
        pdf_dir=PDF_DIRECTORY,
        cache_path=CACHE_FILE_PATH,
        embedding_model=rag_model,
        device=DEVICE
    )

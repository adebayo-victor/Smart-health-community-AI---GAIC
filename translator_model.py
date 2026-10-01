"""
SmartHealth Community AI — Machine Translation Module (NLLB-200)
WOGAFH x GAIC Cohort 2 (Code for Africa AI Sandbox)

File: smarthealth-community-ai/translator_model.py
Responsibility: Loads Meta NLLB-200 (600M) for full bidirectional translation 
across Nigerian Pidgin (pcm_Latn), Hausa (hau_Latn), and Clinical English (eng_Latn).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SmartHealth Translator] Initializing NLLB-200 Translation Engine on: {DEVICE.upper()}")

# Load Meta NLLB-200 Distilled (600M Parameters)
MODEL_ID = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# BCP-47 Language Codes for Meta NLLB
LANG_CODES = {
    "pcm": "pcm_Latn",  # Nigerian Pidgin
    "hau": "hau_Latn",  # Hausa
    "eng": "eng_Latn"   # English
}

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Generic bidirectional translation helper between any two supported dialects.
    """
    if src_lang == tgt_lang:
        return text

    src_tag = LANG_CODES.get(src_lang, "eng_Latn")
    tgt_tag = LANG_CODES.get(tgt_lang, "eng_Latn")
    
    tokenizer.src_lang = src_tag
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=150).to(DEVICE)
    target_lang_id = tokenizer.convert_tokens_to_ids(tgt_tag)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()


# --- SPECIALIZED HELPER FUNCTIONS ---

def translate_to_english(text: str, src_dialect: str) -> str:
    """
    Inbound Helper: Converts Pidgin ('pcm') or Hausa ('hau') into 
    clinical English to maximize RAG semantic search accuracy on English STG PDFs.
    """
    if src_dialect == "pcm":
        return translate(text, src_lang="pcm", tgt_lang="eng")
    elif src_dialect == "hau":
        return translate(text, src_lang="hau", tgt_lang="eng")
    return text


def translate_from_english(english_advice: str, target_dialect: str) -> str:
    """
    Outbound Helper: Converts verified English clinical guidance into 
    natural Nigerian Pidgin ('pcm') or Hausa ('hau') for TTS audio synthesis.
    """
    if target_dialect == "pcm":
        return translate(english_advice, src_lang="eng", tgt_lang="pcm")
    elif target_dialect == "hau":
        return translate(english_advice, src_lang="eng", tgt_lang="hau")
    return english_advice

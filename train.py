"""
SmartHealth AI - Hausa ASR Fine-Tuning Script (Multilingual Upgrade)
Fine-tunes facebook/wav2vec2-xlsr-53 with LoRA on the FLEURS Hausa (Nigeria) dataset.
Optimized for memory-constrained environments (e.g., T4 GPUs).
"""

import torch
import torchaudio
import re
import numpy as np
from dataclasses import dataclass
from datasets import load_dataset, Audio
from transformers import (
    AutoProcessor, 
    AutoModelForCTC, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# ✅ ULTIMATE UPGRADE: XLSR-53 base, already fine-tuned on Hausa, with a working processor!
MODEL_ID = "infinitejoy/wav2vec2-large-xls-r-300m-hausa" 
DATASET_NAME = "google/fleurs"
LANGUAGE_CODE = "ha_ng"  # Hausa - Nigeria
OUTPUT_DIR = "./smarthealth_mms_hau_lora"
MAX_STEPS = 1500  # ✅ UPGRADED: Enough steps to learn from the full dataset

print("🚀 Starting SmartHealth ASR Training Pipeline (Multilingual)...")

# ==============================================================================
# 2. LOAD AND PREPROCESS DATASET
# ==============================================================================
print(f"Loading {DATASET_NAME} ({LANGUAGE_CODE})...")
dataset = load_dataset(DATASET_NAME, LANGUAGE_CODE)

# Cast audio to 16kHz (Crucial for Wav2Vec2)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Clean text: remove special characters and lowercase
chars_to_ignore_regex = r'[^\w\s]' 
def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower()
    return batch

print("Cleaning text...")
dataset = dataset.map(remove_special_characters)

# ✅ UPGRADED: Train on the FULL dataset, no limits!
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"✅ Dataset loaded. Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# ==============================================================================
# 3. LOAD PROCESSOR AND MODEL
# ==============================================================================
print(f"Loading model: {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(MODEL_ID)

# Set CTC specific configs
model.config.vocab_size = len(processor.tokenizer)
model.config.pad_token_id = processor.tokenizer.pad_token_id
print("✅ Model and processor loaded.")

# ==============================================================================
# 4. APPLY LORA (PEFT)
# ==============================================================================
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✅ LoRA applied successfully.")

# ==============================================================================
# 5. DATA PROCESSING FUNCTION
# ==============================================================================
def prepare_dataset(batch):
    # Process audio
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    
    # Process text (FLEURS uses "transcription", not "text")
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

print("Mapping dataset for training...")
train_dataset = train_dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=dataset["test"].column_names)
print("✅ Data processing complete.")

# ==============================================================================
# 6. CUSTOM DATA COLLATOR
# ==============================================================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: bool = True

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding token id with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# ==============================================================================
# 7. EVALUATION METRIC (WER)
# ==============================================================================
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ==============================================================================
# 8. TRAINING ARGUMENTS & TRAINER
# ==============================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,          # Kept at 1 to save VRAM
    gradient_accumulation_steps=16,         # Effective batch size = 16
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    max_steps=MAX_STEPS, 
    fp16=True,                              # Crucial for speed and memory
    gradient_checkpointing=True,            # Saves ~40% VRAM
    eval_strategy="steps",                  
    eval_steps=100,                         # Evaluate every 100 steps
    save_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False, 
    push_to_hub=False,
    report_to="none",
    dataloader_num_workers=0                
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ==============================================================================
# 9. EXECUTE TRAINING
# ==============================================================================
print("🚀 Launching training run...")
trainer.train()

print("\n✅ Training complete! Saving model...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ Model and processor saved to {OUTPUT_DIR}")

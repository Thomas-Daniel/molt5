import pandas as pd
import numpy as np
import torch
import sacrebleu
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)

# --- Config ---
HF_REPO_ID = "nobodytries/smiles_to_captions"
TRAIN_FILE = "train_rag_max_bleu.json"  # Using the max_bleu version you created
MODEL_NAME = "laituan245/molt5-large"
OUTPUT_DIR = "./molt5_bleu_optimized"
BATCH_SIZE = 64
GRADIENT_ACCUMULATION = 4     # 8 * 4 = 32 effective batch size  # H100 allows large batches
EPOCHS = 6       
LR = 3e-5        

# --- 1. Load Data from Hugging Face ---
print(f"Loading {TRAIN_FILE} from {HF_REPO_ID}...")

# Load the specific file as the 'train' split
dataset = load_dataset(HF_REPO_ID, data_files={"train": TRAIN_FILE}, split="train")

# Convert to Pandas to perform "Chemically Safe" splitting
# (We must group by target_text so the same molecule doesn't end up in both Train and Val)
df = dataset.to_pandas()

print(f"Total samples loaded: {len(df)}")

# --- 2. Create Leak-Proof Validation Split ---
# Get unique descriptions (proxy for unique molecules)
unique_descs = df['target_text'].unique()
np.random.seed(42)

# Select 5% of molecules for validation
val_descs = np.random.choice(unique_descs, size=int(len(unique_descs)*0.05), replace=False)

# Split based on description, not random rows
val_df = df[df['target_text'].isin(val_descs)]
train_df = df[~df['target_text'].isin(val_descs)]

print(f"Training Samples: {len(train_df)}")
print(f"Validation Samples: {len(val_df)}")

# Convert back to HF Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# --- 3. Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    # Inputs: "description: <Neighbor> smiles: <Target>"
    model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    
    # Targets: "<Target Description>"
    labels = tokenizer(examples["target_text"], max_length=512, truncation=True, padding="max_length")
    
    # Replace padding with -100 for loss calculation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing...")
train_tok = train_dataset.map(preprocess, batched=True, desc="Tokenizing Train")
val_tok = val_dataset.map(preprocess, batched=True, desc="Tokenizing Val")

# --- 4. Custom Metric (BLEU) ---
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    
    # Decode Generated Text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Decode Reference Text (handling -100)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    
    # Calculate BLEU-4
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    
    return {"bleu": bleu.score}

# --- 5. Training Setup ---
print(f"Loading Model: {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION, # 4
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    
    # CRITICAL: Generate text during validation to calculate BLEU
    predict_with_generate=True,  
    generation_max_length=512,
    generation_num_beams=4,      
    
    bf16=True,                   # Optimized for H100
    load_best_model_at_end=True, 
    metric_for_best_model="bleu",
    greater_is_better=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)

# --- 6. Run ---
print("Starting Training on H100...")
trainer.train()

print(f"Best BLEU Score: {trainer.state.best_metric}")
trainer.save_model("final_best_model")
print("Model saved to ./final_best_model")
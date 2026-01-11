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
# CRITICAL CHANGE: Use the file with separate columns (smiles, neighbor_desc)
# so we can build the RAG prompt dynamically in the script.
TRAIN_FILE = "train_augmented.json" 
MODEL_NAME = "laituan245/molt5-large"
OUTPUT_DIR = "./molt5_rag_final"

# H100 Optimization Settings
BATCH_SIZE = 48
GRADIENT_ACCUMULATION = 1     # Effective Batch Size = 32
EPOCHS = 1       
LR = 3e-5        

# --- 1. Load Data ---
print(f"Loading {TRAIN_FILE} from {HF_REPO_ID}...")
dataset = load_dataset(HF_REPO_ID, data_files={"train": TRAIN_FILE}, split="train")
df = dataset.to_pandas()

print(f"Total samples loaded: {len(df)}")
# Columns Check (Just for sanity)
print(f"Columns found: {df.columns.tolist()}")

# --- 2. Leak-Proof Validation Split ---
# We split by 'target_desc' (or 'id') to ensure no molecule leaks into validation
unique_ids = df['id'].unique()
np.random.seed(42)

# Select 5% of IDs for validation
val_ids = np.random.choice(unique_ids, size=int(len(unique_ids)*0.05), replace=False)

val_df = df[df['id'].isin(val_ids)]
train_df = df[~df['id'].isin(val_ids)]

print(f"Training Samples: {len(train_df)}")
print(f"Validation Samples: {len(val_df)}")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# --- 3. Tokenization & RAG Prompting ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    # --- RAG STYLE PROMPT CONSTRUCTION ---
    # We build the prompt dynamically here.
    # Format: "smiles: <SMILES> description: <NEIGHBOR_DESC>"
    # This aligns the target SMILES with the Neighbor's description as context.
    
    # You can change the string below to experiment with styles:
    # e.g., f"Task: Describe the molecule. Context: {desc} SMILES: {smi}"
    
    inputs = [
        f"smiles: {smi} description: {desc}" 
        for smi, desc in zip(examples["smiles"], examples["neighbor_desc"])
    ]
    
    # Target: The REAL description of the molecule
    targets = examples["target_desc"]
    
    # Tokenize Inputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Tokenize Targets
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    
    # Replace padding with -100 for loss calculation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing with RAG Prompt...")
train_tok = train_dataset.map(preprocess, batched=True, desc="Tokenizing Train")
val_tok = val_dataset.map(preprocess, batched=True, desc="Tokenizing Val")

# --- 4. Custom Metric (BLEU) ---
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Handle -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    return {"bleu": bleu.score}

# --- 5. Training Setup ---
print(f"Loading Model: {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Optimization for H100 VRAM
model.gradient_checkpointing_enable()

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    
    # H100 Memory Optimizations
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_checkpointing=False,
    
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    
    predict_with_generate=True,  
    generation_max_length=512,
    generation_num_beams=2,      
    
    bf16=True, 
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
torch.cuda.empty_cache()
trainer.train()

print(f"Best BLEU Score: {trainer.state.best_metric}")
trainer.save_model("final_best_rag_model")
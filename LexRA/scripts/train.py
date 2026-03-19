import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAIN_FILE   = "data/processed/train.jsonl"
VAL_FILE     = "data/processed/val.jsonl"
OUTPUT_DIR   = "model/checkpoints"
MAX_LENGTH   = 512
EPOCHS       = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 8
LR           = 2e-4

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_data = Dataset.from_list(load_jsonl(TRAIN_FILE))
val_data   = Dataset.from_list(load_jsonl(VAL_FILE))

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

train_data = train_data.map(tokenize, batched=True, remove_columns=["text"])
val_data   = val_data.map(tokenize,   batched=True, remove_columns=["text"])

# ── Quantization Config (4-bit) ───────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ── Load Model ────────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# ── LoRA Config ───────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training Args ─────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR + "/best_model")
tokenizer.save_pretrained(OUTPUT_DIR + "/best_model")
print("Model saved to", OUTPUT_DIR + "/best_model")
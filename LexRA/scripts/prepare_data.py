# from datasets import load_dataset

# print(load_dataset("viber1/indian-law-dataset"))

from datasets import load_dataset
import json
import os

# Load dataset
dataset = load_dataset("viber1/indian-law-dataset")
data = dataset["train"]

# Format into prompt template
def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['Instruction']}\n\n### Response:\n{example['Response']}"
    }

data = data.map(format_prompt)

# Train / val split (90/10)
split = data.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
val_data = split["test"]

# Save to processed/
os.makedirs("data/processed", exist_ok=True)

def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for example in dataset:
            json.dump({"text": example["text"]}, f, ensure_ascii=False)
            f.write("\n")

save_jsonl(train_data, "data/processed/train.jsonl")
save_jsonl(val_data,   "data/processed/val.jsonl")

print(f"Train samples : {len(train_data)}")
print(f"Val samples   : {len(val_data)}")
print("Saved to data/processed/")
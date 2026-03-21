import json
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINETUNED_MODEL = "aapnakaamkar/LexRA-TinyLlama-Legal"  # path to your checkpoint
VAL_FILE       = "data/processed/val.jsonl"
MAX_LENGTH     = 512
NUM_SAMPLES    = 100  # evaluate on 100 val samples

# ── Load Val Data ─────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]

val_texts = load_jsonl(VAL_FILE)[:NUM_SAMPLES]
print(f"Evaluating on {len(val_texts)} samples")

# ── Perplexity Function ───────────────────────────────────────────────────────
def compute_perplexity(model, tokenizer, texts):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            )
            input_ids = inputs["input_ids"]

            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item()
            count += 1

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    return round(perplexity, 4), round(avg_loss, 4)

# ── Load Tokenizer ────────────────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ── Evaluate Base Model ───────────────────────────────────────────────────────
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Evaluating base model...")
base_ppl, base_loss = compute_perplexity(base_model, tokenizer, val_texts)
print(f"Base Model     → Loss: {base_loss} | Perplexity: {base_ppl}")

del base_model  # free memory

# ── Evaluate Finetuned Model ──────────────────────────────────────────────────
print("\nLoading finetuned model...")
ft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)
ft_model = PeftModel.from_pretrained(ft_model, FINETUNED_MODEL)
ft_model = ft_model.merge_and_unload()  # merge LoRA into base

print("Evaluating finetuned model...")
ft_ppl, ft_loss = compute_perplexity(ft_model, tokenizer, val_texts)
print(f"Finetuned Model → Loss: {ft_loss} | Perplexity: {ft_ppl}")

# ── Results ───────────────────────────────────────────────────────────────────
improvement = round(((base_ppl - ft_ppl) / base_ppl) * 100, 2)
print(f"\n── Results ───────────────────────────────")
print(f"Base Perplexity      : {base_ppl}")
print(f"Finetuned Perplexity : {ft_ppl}")
print(f"Improvement          : {improvement}%")

# Save results
import json
results = {
    "base_loss": base_loss,
    "base_perplexity": base_ppl,
    "finetuned_loss": ft_loss,
    "finetuned_perplexity": ft_ppl,
    "improvement_percent": improvement
}
with open("docs/perplexity_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to docs/perplexity_results.json")
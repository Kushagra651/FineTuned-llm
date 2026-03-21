import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINETUNED_MODEL = "aapnakaamkar/LexRA-TinyLlama-Legal"
MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.3

# ── Load Tokenizer ────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ── Load Base Model ───────────────────────────────────────────────────────────
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# ── Merge LoRA Adapter ────────────────────────────────────────────────────────
print("Merging LoRA adapter...")
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
model = model.merge_and_unload()
model.eval()
print("Model ready!\n")

# ── Pipeline ──────────────────────────────────────────────────────────────────
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    do_sample=True,
    repetition_penalty=1.2
)

# ── Inference Function ────────────────────────────────────────────────────────
def generate(instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    output = pipe(prompt)[0]["generated_text"]
    # Extract only the response part
    response = output.split("### Response:\n")[-1].strip()
    return response

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is the meaning of bail in Indian law?",
        "Explain the concept of anticipatory bail.",
        "What are the fundamental rights under the Indian Constitution?"
    ]

    for query in test_queries:
        print(f"Query    : {query}")
        print(f"Response : {generate(query)}")
        print("-" * 60)
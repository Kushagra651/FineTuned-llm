import torch
import gradio as gr
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

# ── Load Base Model Pipeline ──────────────────────────────────────────────────
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)
base_pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    do_sample=True,
    repetition_penalty=1.2
)

# ── Load Finetuned Model Pipeline ─────────────────────────────────────────────
print("Loading finetuned model...")
ft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)
ft_model = PeftModel.from_pretrained(ft_model, FINETUNED_MODEL)
ft_model = ft_model.merge_and_unload()
ft_model.eval()
ft_pipe = pipeline(
    "text-generation",
    model=ft_model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    do_sample=True,
    repetition_penalty=1.2
)
print("Both models ready!\n")

# ── Generate Function ─────────────────────────────────────────────────────────
def generate(instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    base_out = base_pipe(prompt)[0]["generated_text"]
    base_response = base_out.split("### Response:\n")[-1].strip()

    ft_out = ft_pipe(prompt)[0]["generated_text"]
    ft_response = ft_out.split("### Response:\n")[-1].strip()

    return base_response, ft_response

# ── Gradio UI ─────────────────────────────────────────────────────────────────
examples = [
    "What is the meaning of bail in Indian law?",
    "Explain the concept of anticipatory bail.",
    "What are fundamental rights under the Indian Constitution?",
    "What is the difference between cognizable and non-cognizable offences?",
]

with gr.Blocks(title="LexRA — Legal Reasoning Assistant") as demo:

    gr.Markdown("""
    # ⚖️ LexRA — Legal Reasoning Assistant
    ### TinyLlama 1.1B fine-tuned on 22,000+ Indian Supreme Court Judgements
    Compare **Base TinyLlama** vs **Fine-tuned LexRA** on Indian legal queries.
    """)

    with gr.Row():
        query = gr.Textbox(
            label="Enter your legal query",
            placeholder="e.g. What is anticipatory bail?",
            lines=3
        )

    submit_btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        base_output = gr.Textbox(
            label="❌ Base TinyLlama (no legal knowledge)",
            lines=10,
            interactive=False
        )
        ft_output = gr.Textbox(
            label="✅ LexRA Fine-tuned (trained on Indian legal data)",
            lines=10,
            interactive=False
        )

    gr.Examples(examples=examples, inputs=query)

    submit_btn.click(
        fn=generate,
        inputs=query,
        outputs=[base_output, ft_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
# ⚖️ LexRA — Legal Reasoning Assistant

> Fine-tuned TinyLlama 1.1B on 22,000+ Indian Supreme Court Judgements using QLoRA for domain-specific legal language understanding.

[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/aapnakaamkar/LexRA-TinyLlama-Legal)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/aapnakaamkar/LexRA)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

---

## 📌 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Core Concepts](#core-concepts)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Resume Highlights](#resume-highlights)
- [References](#references)

---

## 🔍 Overview

LexRA is an end-to-end LLM finetuning project that adapts **TinyLlama 1.1B** to Indian legal language using **QLoRA** — the industry-standard technique for efficient finetuning on limited hardware.

| Property | Value |
|----------|-------|
| Base Model | TinyLlama 1.1B Chat |
| Dataset | Indian Supreme Court Judgements |
| Training Samples | 22,146 |
| Finetuning Method | QLoRA (4-bit NF4 + LoRA) |
| Trainable Parameters | 6.3M / 1.1B (0.57%) |
| Training Hardware | Google Colab T4 GPU (16GB) |
| Base Perplexity | 6.51 |
| Finetuned Perplexity | 3.07 |
| Improvement | **52.84%** |

---

## 🎯 Demo

**Live Demo:** [huggingface.co/spaces/aapnakaamkar/LexRA](https://huggingface.co/spaces/aapnakaamkar/LexRA)

**Model Weights:** [huggingface.co/aapnakaamkar/LexRA-TinyLlama-Legal](https://huggingface.co/aapnakaamkar/LexRA-TinyLlama-Legal)

---

## ❗ Problem Statement

General-purpose language models perform poorly on domain-specific legal text because:
- Legal language has highly specialized vocabulary (appellant, petitioner, writ, cognizable)
- Indian legal documents follow specific structural patterns different from general English
- Full finetuning of 1.1B+ parameter models requires 14GB+ GPU memory — inaccessible on free hardware

---

## ✅ Solution

1. Load TinyLlama in **4-bit NF4 quantization** — reduces memory from 4.4GB to 550MB
2. Inject **LoRA adapters** (rank 8) — only 0.57% of parameters trained
3. Train on **22,000 Indian Supreme Court judgements** as instruction-response pairs
4. Evaluate using **perplexity** — 52.84% improvement over base model
5. Deploy via **Gradio on HuggingFace Spaces** with side-by-side comparison

---

## 📊 Key Results

```
Metric                  Base TinyLlama    LexRA Finetuned    Improvement
-------------------------------------------------------------------------
Perplexity              6.51              3.07               52.84% ↓
Cross Entropy Loss      1.874             1.121              40.18% ↓
Trainable Parameters    1.1B              6.3M               99.43% ↓
Model Memory (GPU)      4.4 GB            0.55 GB            87.5%  ↓
Adapter File Size       —                 ~50 MB             —
```

---

## 📁 Project Structure

```
LexRA/
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.jsonl             # 22,146 samples
│       └── val.jsonl               # 2,461 samples
├── scripts/
│   ├── prepare_data.py
│   └── evaluate.py
├── app/
│   ├── app.py
│   └── inference.py
├── notebooks/
│   └── train_collab.ipynb
├── docs/
│   ├── perplexity_results.json
│   └── comparison_screenshot.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.10 | Core development |
| Deep Learning | PyTorch | Tensor operations, GPU compute |
| LLM Framework | HuggingFace Transformers | Model loading, tokenization, training |
| Efficient Finetuning | PEFT | LoRA adapter injection and management |
| Quantization | BitsAndBytes | 4-bit NF4 quantization |
| Training Loop | HuggingFace Trainer | Automated training with checkpointing |
| Data Processing | HuggingFace Datasets | Dataset loading and preprocessing |
| Mixed Precision | Accelerate | fp16 training on GPU |
| Frontend | Gradio | Web interface for inference |
| Model Hosting | HuggingFace Hub | Adapter weights storage |
| Deployment | HuggingFace Spaces | Public inference API with auto Docker CI/CD |
| Training Hardware | Google Colab T4 | Free 16GB GPU |

---

## 📚 Core Concepts

### QLoRA — Quantized Low Rank Adaptation

QLoRA combines quantization and LoRA to make finetuning large models on limited hardware possible.

**Q = Quantization (4-bit NF4)**
```
float32:  1.1B × 4 bytes  = 4.4 GB
NF4 4bit: 1.1B × 0.5 byte = 0.55 GB  (87.5% reduction)
```

**LoRA = Low Rank Adaptation**
```
Instead of updating W (4096×4096 = 16.7M params):
Learn A (4096×8) + B (8×4096) = 65K params

Output = W×x + B×A×x
W stays frozen. Only A and B are trained.
```

### NF4 — NormalFloat 4-bit

Neural network weights follow a normal distribution — most values cluster near zero. NF4 places its 16 quantization buckets based on this distribution, concentrating precision where most weights exist.

```
Regular 4-bit: equal bucket spacing  → more rounding error
NF4:           normal dist spacing   → less rounding error for LLM weights
```

### Perplexity

Measures how "confused" the model is predicting each next token. Lower = better.

```
Perplexity = e^(cross_entropy_loss)

Base:       e^1.874 = 6.51  (confused between ~6 words per step)
Finetuned:  e^1.121 = 3.07  (confused between ~3 words per step)
```

### Projection Layers

Linear transformations inside attention that map input to different representation spaces:
```
q_proj → Query:  what am I looking for?
k_proj → Key:    what do I contain?
v_proj → Value:  what do I return if selected?
```
LoRA targets q_proj and v_proj — the most impactful layers for domain adaptation.

### Gradient Accumulation

Simulates larger batch sizes on limited GPU memory:
```
batch_size=2 × gradient_accumulation=8 → effective batch = 16
Accumulate gradients for 16 samples before one weight update
Same result as batch_size=16 at 8× less memory
```

### LoRA Adapter Initialization

```
A → random Gaussian initialization
B → zero initialization

At start: B×A = 0, model behaves exactly like base model
As training progresses: B learns, corrections gradually applied
```

B is initialized to zero (not A) so gradients flow properly to both matrices from step 1.

---

## 📦 Dataset

**Source:** [viber1/indian-law-dataset](https://huggingface.co/datasets/viber1/indian-law-dataset)

| Property | Value |
|----------|-------|
| Total Samples | 24,607 |
| Train Split | 22,146 (90%) |
| Validation Split | 2,461 (10%) |
| Columns | Instruction, Response |
| Domain | Indian Supreme Court Judgements |

**Prompt Format:**
```
### Instruction:
What is the meaning of anticipatory bail?

### Response:
Anticipatory bail is a direction to release a person on bail
issued even before the person is arrested...
```

---

## 🏗️ Model Architecture

```
Input → Tokenizer → Embeddings
                        ↓
            [Transformer Block × 22]
                        │
            ┌───────────┴───────────┐
            │                       │
        W×x (frozen NF4)        B×A×x (LoRA, trainable)
            │                       │
            └───────────┬───────────┘
                    Addition
                        ↓
                [Next Layer]
                        ↓
            Language Model Head
                        ↓
            Token Probabilities → Text
```

**LoRA Config:**
```python
LoraConfig(r=8, lora_alpha=16,
           target_modules=["q_proj", "v_proj"],
           lora_dropout=0.05, bias="none",
           task_type="CAUSAL_LM")
```

---

## 🚀 Training Pipeline

```
HuggingFace Dataset
        ↓ prepare_data.py
JSONL train/val files
        ↓ train_collab.ipynb
Tokenize (max_length=512)
        ↓
Load TinyLlama in 4-bit NF4
        ↓
prepare_model_for_kbit_training()
        ↓
get_peft_model() → 0.57% trainable params
        ↓
HuggingFace Trainer (3 epochs, lr=2e-4, fp16)
Save checkpoint every 500 steps
        ↓
Best model → HuggingFace Hub
```

---

## 📈 Evaluation

```bash
python scripts/evaluate.py
```

Output:
```
Base Model      → Loss: 1.8740 | Perplexity: 6.5187
Finetuned Model → Loss: 1.1210 | Perplexity: 3.0745

Improvement: 52.84%
Saved to docs/perplexity_results.json
```

---

## ⚙️ Installation

```bash
git clone https://github.com/aapnakaamkar/LexRA.git
cd LexRA
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**requirements.txt:**
```
torch
transformers
peft
datasets
gradio
accelerate
bitsandbytes>=0.46.1
ipykernel
```

---

## 💻 Usage

**Prepare data:**
```bash
python scripts/prepare_data.py
```

**Train (Colab):**
1. Upload `train_collab.ipynb` to Google Colab
2. Runtime → T4 GPU
3. Run all cells

**Evaluate:**
```bash
python scripts/evaluate.py
```

**Run app:**
```bash
python app/app.py
# http://127.0.0.1:7860
```

**Load model in code:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                              torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, "aapnakaamkar/LexRA-TinyLlama-Legal")
model = model.merge_and_unload()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=256, temperature=0.3, do_sample=True)
prompt = "### Instruction:\nWhat is bail?\n\n### Response:\n"
print(pipe(prompt)[0]["generated_text"].split("### Response:\n")[-1])
```

---

## 🌐 Deployment

```
Upload app.py + requirements.txt to HuggingFace Space
                    ↓
HF detects Gradio SDK → builds Docker container
                    ↓
Installs requirements → runs app.py
                    ↓
Serves public HTTPS URL (auto-rebuilds on file change)
```

**URL:** https://huggingface.co/spaces/aapnakaamkar/LexRA

---

## 🏆 Resume Highlights

- Fine-tuned TinyLlama 1.1B on 22,000+ Indian Supreme Court judgements using LoRA adapters, reducing trainable parameters by 94% from 1.1B to 6.3M
- Implemented 4-bit NF4 quantization via BitsAndBytes reducing model memory from 4.4GB to 550MB enabling training on free-tier T4 GPU
- Engineered data preprocessing pipeline to format 24,000+ raw legal records into instruction-response pairs for supervised finetuning
- Achieved 52.84% perplexity reduction (6.51 → 3.07) on held-out Indian legal text demonstrating successful domain adaptation
- Deployed interactive Gradio interface on HuggingFace Spaces showcasing side-by-side base vs finetuned model comparison on Indian legal queries

---

## 📖 References

| Resource | Link |
|----------|------|
| QLoRA Paper (Dettmers et al. 2023) | [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314) |
| LoRA Paper (Hu et al. 2021) | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
| TinyLlama | [huggingface.co/TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| Dataset | [viber1/indian-law-dataset](https://huggingface.co/datasets/viber1/indian-law-dataset) |
| PEFT Library | [github.com/huggingface/peft](https://github.com/huggingface/peft) |
| BitsAndBytes | [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) |
| HuggingFace Transformers | [github.com/huggingface/transformers](https://github.com/huggingface/transformers) |
| Attention Is All You Need | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Kushagra Bhargava**
- HuggingFace: [aapnakaamkar](https://huggingface.co/aapnakaamkar)
- GitHub: [kushagra651]([https://github.com/Kushagra651/](https://github.com/Kushagra651/))

---
*Built with QLoRA, PEFT, and the HuggingFace ecosystem*

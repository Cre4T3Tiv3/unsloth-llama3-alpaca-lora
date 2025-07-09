<p align="center">
  <img src="https://raw.githubusercontent.com/Cre4T3Tiv3/unsloth-llama3-alpaca-lora/main/docs/assets/unsloth_llama3_alpaca_lora_v0.1.0.png" alt="Demo GIF" width="640"/>
</p>

<p align="center">
  <i>Instruction-tuned LoRA adapter for LLaMA 3 8B using QLoRA + Alpaca-style prompts, trained with Unsloth.</i>
</p>

<p align="center">
  <a href="https://huggingface.co/Cre4T3Tiv3/unsloth-llama3-alpaca-lora">
    <img src="https://img.shields.io/badge/HF_Model-Available-blue?logo=huggingface" alt="HF Model">
  </a>
  <a href="https://huggingface.co/spaces/Cre4T3Tiv3/unsloth-llama3-demo">
    <img src="https://img.shields.io/badge/Live_Demo-HF_Space-orange?logo=gradio" alt="HF Demo Space">
  </a>
  <a href="https://github.com/Cre4T3Tiv3/unsloth-llama3-alpaca-lora/stargazers">
    <img src="https://img.shields.io/github/stars/Cre4T3Tiv3/unsloth-llama3-alpaca-lora?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/Cre4T3Tiv3/unsloth-llama3-alpaca-lora/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Cre4T3Tiv3/unsloth-llama3-alpaca-lora" alt="License">
  </a>
  <a href="https://bytestacklabs.com">
    <img src="https://img.shields.io/badge/Made%20by-ByteStack%20Labs-2ea44f" alt="ByteStack Labs">
  </a>
</p>

---

## Overview

This repo hosts the training, evaluation, and inference pipeline for:

> **`Cre4T3Tiv3/unsloth-llama3-alpaca-lora`**

A 4-bit QLoRA LoRA adapter trained on:

- [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- 30+ grounded examples of QLoRA reasoning (added to mitigate hallucinations)

### Core Stack

- **Base Model**: `unsloth/llama-3-8b-bnb-4bit`
- **Adapter Format**: LoRA (merged post-training)
- **Training Framework**: [Unsloth](https://github.com/unslothai/unsloth) + HuggingFace PEFT
- **Training Infra**: A100 (40GB), 4-bit quantization

---

## Intended Use

This adapter is purpose-built for:

- Instruction-following LLM tasks
- Low-resource, local inference (4-bit, merged LoRA)
- Agentic tools and CLI assistants
- Educational demos (fine-tuning, PEFT, Unsloth)
- Quick deployment in QLoRA-aware stacks

### Limitations

- Trained on ~2K samples + 3 custom prompts
- Single-run fine-tune only
- Not optimized for >2K context
- 4-bit quantization may reduce fidelity
- Hallucinations possible — **not** production-ready for critical workflows
- Previously hallucinated QLoRA terms now corrected — tested via eval script
- Still not production-grade for factual QA or critical domains

---

## Evaluation

This repo includes an `eval_adapter.py` script that:

- Checks for hallucination patterns (e.g. false QLoRA definitions)
- Computes keyword overlap per instruction (≥4/6 threshold)
- Outputs JSON summary (`eval_results.json`) with full logs

> Run `make eval` to validate adapter behavior.

---

## Training Configuration

| Parameter       | Value                               |
|-----------------|-------------------------------------|
| Base Model      | `unsloth/llama-3-8b-bnb-4bit`       |
| Adapter Format  | LoRA (merged)                       |
| LoRA `r`        | 16                                  |
| LoRA `alpha`    | 16                                  |
| LoRA `dropout`  | 0.05                                |
| Epochs          | 2                                   |
| Examples        | ~2K (alpaca-cleaned + grounded)     |
| Precision       | 4-bit (bnb)                         |

---

## Usage

```bash
make install   # Create .venv and install with uv
make train     # Train LoRA adapter
make eval      # Evaluate output quality
make run       # Run quick inference
````

### Hugging Face Login

```bash
export HUGGINGFACE_TOKEN=hf_xxx
make login
```

---

## Local Inference (Python)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "unsloth/llama-3-8b-bnb-4bit"
ADAPTER = "Cre4T3Tiv3/unsloth-llama3-alpaca-lora"

base_model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(base_model, ADAPTER).merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

prompt = "### Instruction:\nExplain LoRA fine-tuning in simple terms.\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Demo Space

🖥 Try the model live via Hugging Face Spaces:

> [Launch Demo → unsloth-llama3-demo](https://huggingface.co/spaces/Cre4T3Tiv3/unsloth-llama3-demo)

---

## Links

* 📦 [Model Hub](https://huggingface.co/Cre4T3Tiv3/unsloth-llama3-alpaca-lora)
* 🧪 [Demo Space](https://huggingface.co/spaces/Cre4T3Tiv3/unsloth-llama3-demo)
* 🧰 [Source Code](https://github.com/Cre4T3Tiv3/unsloth-llama3-alpaca-lora)
* 💼 [ByteStack Labs](https://bytestacklabs.com)

---

## Built With

* [Unsloth](https://github.com/unslothai/unsloth)
* [Transformers](https://github.com/huggingface/transformers)
* [PEFT](https://github.com/huggingface/peft)
* [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

## Maintainer

**[@Cre4T3Tiv3](https://github.com/Cre4T3Tiv3)**
Built with ❤️ by [ByteStack Labs](https://bytestacklabs.com)

---

## Citation

If you use this adapter or its training methodology, please consider citing:

```
@software{unsloth-llama3-alpaca-lora,
  author = {Jesse Moses, Cre4T3Tiv3},
  title = {Unsloth LoRA Adapter for LLaMA 3 (8B)},
  year = {2025},
  url = {https://huggingface.co/Cre4T3Tiv3/unsloth-llama3-alpaca-lora},
}
```

---

## License

Apache 2.0

---


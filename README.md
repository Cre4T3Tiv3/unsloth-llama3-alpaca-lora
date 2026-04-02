4-bit QLoRA fine-tuning pipeline for LLaMA 3 8B. From training configuration to published adapter on HuggingFace. Memory-efficient instruction tuning on consumer GPUs using Unsloth.

<p align="center">
  <a href="https://github.com/Cre4T3Tiv3/unsloth-llama3-alpaca-lora" target="_blank">
    <img src="https://raw.githubusercontent.com/Cre4T3Tiv3/unsloth-llama3-alpaca-lora/main/docs/assets/unsloth_llama3_alpaca_lora_v0.1.0_latest.png" alt="Unsloth LLaMA 3 Alpaca LoRA" width="640"/>
  </a>
</p>

<p align="center">
  <a href="https://huggingface.co/Cre4T3Tiv3/unsloth-llama3-alpaca-lora">
    <img src="https://img.shields.io/badge/HF_Model-Available-blue?logo=huggingface" alt="HF Model">
  </a>
  <a href="https://huggingface.co/spaces/Cre4T3Tiv3/unsloth-llama3-alpaca-demo">
    <img src="https://img.shields.io/badge/Live_Demo-HF_Space-orange?logo=gradio" alt="HF Demo">
  </a>
  <a href="https://github.com/Cre4T3Tiv3/unsloth-llama3-alpaca-lora/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Cre4T3Tiv3/unsloth-llama3-alpaca-lora" alt="License">
  </a>
  <a href="https://bytestacklabs.com">
    <img src="https://img.shields.io/badge/Made%20by-ByteStack%20Labs-2ea44f" alt="ByteStack Labs">
  </a>
</p>

---

## What This Is

An end-to-end pipeline for custom model training: dataset preparation, QLoRA fine-tuning, evaluation, and deployment. The adapter is trained on the Alpaca-cleaned instruction dataset plus grounded QLoRA reasoning examples added to mitigate hallucinations. Published and runnable on HuggingFace.

## Training Configuration

| Parameter | Value |
| --- | --- |
| Base Model | `unsloth/llama-3-8b-bnb-4bit` |
| Adapter Format | LoRA (merged post-training) |
| LoRA r / alpha / dropout | 16 / 16 / 0.05 |
| Epochs | 2 |
| Training Data | ~2K examples (alpaca-cleaned + grounded) |
| Precision | 4-bit (bitsandbytes) |
| Training Hardware | A100 (40GB) |
| Framework | Unsloth + HuggingFace PEFT |

## Evaluation

The included `eval_adapter.py` script checks for hallucination patterns (false QLoRA definitions), computes keyword overlap per instruction against a threshold, and outputs a JSON summary. Run `make eval` to validate adapter behavior.

## Usage

```bash
make install   # Create .venv and install with uv
make train     # Train LoRA adapter
make eval      # Evaluate output quality
make run       # Run inference
```

### Local Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit", device_map="auto", load_in_4bit=True
)
model = PeftModel.from_pretrained(
    base_model, "Cre4T3Tiv3/unsloth-llama3-alpaca-lora"
).merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("Cre4T3Tiv3/unsloth-llama3-alpaca-lora")

prompt = "### Instruction:\nExplain LoRA fine-tuning in simple terms.\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

Trained on ~2K samples in a single fine-tuning run. Not optimized for contexts longer than 2K tokens. 4-bit quantization may reduce fidelity. Not production-grade for factual QA or critical domains.

## Links

- [Model on HuggingFace](https://huggingface.co/Cre4T3Tiv3/unsloth-llama3-alpaca-lora)
- [Live Demo](https://huggingface.co/spaces/Cre4T3Tiv3/unsloth-llama3-alpaca-demo)

## License

[MIT](LICENSE)

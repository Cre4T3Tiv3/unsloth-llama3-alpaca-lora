"""Evaluation script for a LoRA adapter on LLaMA 3 (8B) using Alpaca-style prompts.

This script:
- Loads a quantized LLaMA 3 8B model via Unsloth.
- Applies a fine-tuned LoRA adapter.
- Evaluates model output against expected keywords and hallucination triggers.
- Retries generation if keyword match is low but no hallucination is detected.
- Outputs structured evaluation results to `eval_results.json`.

Typical usage:
    $ python scripts/eval_adapter.py
    $ make eval
"""

import json
import random
import re

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

RETRY_THRESHOLD = 2
MAX_RETRIES = 2

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

adapter_id = "Cre4T3Tiv3/unsloth-llama3-alpaca-lora"
base_id = "unsloth/llama-3-8b-bnb-4bit"

base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_id, force_download=True)

model = PeftModel.from_pretrained(
    base_model,
    adapter_id,
    force_download=True,
)
model.print_trainable_parameters()
model = model.merge_and_unload()
model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    eos_token_id=tokenizer.eos_token_id,
)


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Formats an Alpaca-style prompt from an instruction and optional input.

    Args:
        instruction: Instruction text for the task.
        input_text: Optional supporting context or question.

    Returns:
        A formatted prompt string.
    """
    if input_text.strip():
        return (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes "
            "the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n### Response:"
        )
    else:
        return (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:"
        )


fuzzy_terms = [
    "quantum",
    "quanta",
    "loosely-attached",
    "rationalization",
    "gpt-3",
    "chatgpt api",
    "chatgpt",
    "openai",
]


def has_hallucination(output: str, strict_terms=None, fuzzy_terms=None) -> bool:
    """Checks if the output contains known hallucination triggers.

    Args:
        output: The generated response string.
        strict_terms: List of explicit hallucination terms from the example definition.
        fuzzy_terms: Global fuzzy terms to catch soft drift.

    Returns:
        Boolean indicating hallucination presence.
    """
    strict_terms = strict_terms or []
    fuzzy_terms = fuzzy_terms or []
    text = output.lower()
    return any(term.lower() in text for term in strict_terms + fuzzy_terms)


examples = [
    {
        "instruction": "Explain the concept of transformer models in deep learning.",
        "input": "Describe how transformers differ from RNNs and why they're effective for language tasks.",
        "expected_keywords": [
            "self-attention",
            "parallel",
            "sequence",
            "rnn",
            "context",
            "long-range",
        ],
    },
    {
        "instruction": "List three advantages of using LLaMA 3 models over traditional language models.",
        "input": "Focus on improvements in efficiency, accuracy, and scalability.",
        "expected_keywords": [
            "efficient",
            "accurate",
            "scalable",
            "lighter",
            "faster",
            "token",
        ],
    },
    {
        "instruction": "Write a tweet that explains QLoRA in simple terms to beginners.",
        "input": "Target AI enthusiasts who are new to fine-tuning techniques.",
        "expected_keywords": [
            "4-bit",
            "quantized",
            "LoRA",
            "efficient",
            "fine-tune",
            "low-resource",
        ],
        "hallucination_terms": [
            "quantized linear regression",
            "quantized regression",
            "quantum",
            "loosely-attached",
            "quantum loosely-attached rationalization",
        ],
    },
]

if torch.cuda.is_available():
    print(f"Device set to use {torch.cuda.get_device_name(0)}")
print(f"Device set to use {'cuda:0' if torch.cuda.is_available() else 'cpu'}\n")

print("📊 Evaluation Results:\n" + "-" * 40)

results = []

for i, ex in enumerate(examples):
    attempt = 0
    final_output = None
    final_hits = 0
    hallucinated = False
    response_len = 0

    while attempt <= MAX_RETRIES:
        prompt = format_prompt(ex["instruction"], ex["input"])
        print(
            f"\n\033[1mExample {i+1} - Attempt {attempt+1}:\033[0m\n{prompt.strip()}\n"
        )

        output = pipe(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
        )[0]["generated_text"].strip()

        output_clean = re.split(r"\n?###\s|\n?Instruction:|\n?Response:", output)[
            0
        ].strip()

        hits = sum(
            1
            for kw in ex.get("expected_keywords", [])
            if kw.lower() in output_clean.lower()
        )

        hallucinated = has_hallucination(
            output_clean,
            ex.get("hallucination_terms", []),
            fuzzy_terms=fuzzy_terms,
        )

        response_len = len(output_clean.split())

        print(f"\033[1m🔹 Model Response:\033[0m\n{output_clean}\n")
        print(
            f"\033[94m✅ Keywords matched:\033[0m {hits} / {len(ex.get('expected_keywords', []))}"
        )
        print(
            f"\033[93m🚫 Hallucination detected:\033[0m {'Yes' if hallucinated else 'No'}"
        )
        print(f"\033[92m📏 Response length:\033[0m {response_len} tokens")

        if hallucinated:
            print(f"\033[91m❌ REJECTED: Hallucination present\033[0m")
            break

        if hits >= RETRY_THRESHOLD or attempt == MAX_RETRIES:
            final_output = output_clean
            final_hits = hits
            break

        print(
            f"\033[90m↩️  Retrying due to low keyword match ({hits}) and no hallucination...\033[0m"
        )
        attempt += 1

    print("=" * 60 + "\n")

    results.append(
        {
            "example": i + 1,
            "instruction": ex["instruction"],
            "input": ex["input"],
            "output": final_output,
            "matched_keywords": final_hits,
            "hallucinated": hallucinated,
            "length": response_len,
            "score": round(final_hits / len(ex.get("expected_keywords", [])), 2),
            "accepted": final_hits >= RETRY_THRESHOLD and not hallucinated,
        }
    )

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

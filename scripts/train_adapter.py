"""Fine-tune a LLaMA 3 8B model using Unsloth and Alpaca-cleaned data with QLoRA.

This script loads the `unsloth/llama-3-8b-bnb-4bit` base model with 4-bit quantization,
applies LoRA adapters, and fine-tunes on a subset of the `yahma/alpaca-cleaned` dataset.

The training is configured to save LoRA adapter weights to `./adapter` using Hugging Face's Trainer API.

Run this script with:
    $ python scripts/train_adapter.py
"""

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel


def main():
    """Runs the LoRA fine-tuning pipeline using Unsloth + Hugging Face Trainer.

    This function performs the following:
    - Loads the base LLaMA 3 8B model with 4-bit quantization
    - Applies LoRA adapter configuration (r=16, alpha=16, dropout=0.05)
    - Loads and tokenizes 2000 examples from the `alpaca-cleaned` dataset
    - Sets up training arguments and trains the adapter
    """

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    data = load_dataset("yahma/alpaca-cleaned")
    dataset = data["train"].shuffle(seed=42).select(range(2000))

    def format(example):
        """Formats a single Alpaca-style example for instruction tuning.

        Args:
            example (dict): A dictionary with `instruction`, `input`, and `output` fields.

        Returns:
            str: A formatted string combining instruction, input, and output sections.
        """
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

    def tokenize(example):
        """Tokenizes and prepares a single example for supervised fine-tuning.

        Args:
            example (dict): A formatted instruction example.

        Returns:
            dict: A dictionary with input token IDs and corresponding labels.
        """
        formatted = format(example)
        tokenized = tokenizer(
            formatted, truncation=True, padding="max_length", max_length=2048
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(tokenize, batched=False)

    training_args = TrainingArguments(
        output_dir="./adapter",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        save_steps=200,
        logging_steps=50,
        learning_rate=2e-4,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()

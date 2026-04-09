from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Optional

VECTOR_STORE_PATH = "rag/vector_store"
DATASET_PATH      = "finetune_data/tutor_dataset.jsonl"
LEVELS            = ["beginner", "intermediate", "advanced"]

# Templates for synthetic question generation
_QUESTION_TEMPLATES = [
    "Explain {concept}.",
    "What is {concept}?",
    "How does {concept} work?",
    "Why is {concept} important?",
    "Give a step-by-step explanation of {concept}.",
    "What are common misconceptions about {concept}?",
    "Compare {concept} with related ideas.",
    "Provide a real-world example of {concept}.",
]


def _extract_key_concept(chunk: str) -> str:
   #First 6 words of chunk, could use LLM technique
    words = chunk.split()[:6]
    return " ".join(words).strip(".,;:\"'")


def generate_dataset_from_corpus(
    n_samples: int = 500,
    output_path: str = DATASET_PATH,
) -> list[dict]:

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(f"{VECTOR_STORE_PATH}/texts.pkl", "rb") as f:
        texts: list[str] = pickle.load(f)

    try:
        from llm.tutor_model import _chat, _TUTOR_SYSTEM, build_prompt
        llm_available = True
    except Exception:
        llm_available = False
        print("[Dataset] LLM not available — generating template-only dataset")

    samples: list[dict] = []
    sampled_chunks = random.sample(texts, min(n_samples, len(texts)))

    print(f"[Dataset] Generating {len(sampled_chunks)} samples…")

    for i, chunk in enumerate(sampled_chunks):
        if len(chunk) < 100:
            continue

        concept  = _extract_key_concept(chunk)
        template = random.choice(_QUESTION_TEMPLATES)
        query    = template.format(concept=concept)
        level    = random.choice(LEVELS)

        if llm_available:
            try:
                answer = _chat(
                    system=_TUTOR_SYSTEM,
                    user=build_prompt(query, chunk, level),
                    max_tokens=700,
                    temperature=0.6,
                )
            except Exception as e:
                print(f"  [Sample {i}] LLM error: {e} — using chunk as answer")
                answer = chunk
        else:
            answer = (
                f"**Topic:** {concept}\n\n"
                f"**Explanation ({level} level):**\n{chunk}\n\n"
                "**Summary:** See above."
            )

        samples.append({
            "instruction": build_prompt(query, chunk, level) if llm_available
                           else f"Explain: {query}\nContext: {chunk}\nLevel: {level}",
            "input":  "",
            "output": answer,
            "level":  level,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sampled_chunks)} samples done")

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Dataset] Saved {len(samples)} samples → {output_path}")
    return samples

BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_OUTPUT  = "finetune_data/tutor_lora_adapter"
HF_CACHE        = os.environ.get("HF_HOME", "~/.cache/huggingface")


def _format_alpaca(sample: dict) -> str:
    """Convert a dataset row to an Alpaca-style prompt string."""
    if sample.get("input", "").strip():
        return (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}"
    )


def train_qlora(
    dataset_path: str  = DATASET_PATH,
    base_model: str    = BASE_MODEL,
    output_dir: str    = ADAPTER_OUTPUT,
    num_epochs: int    = 3,
    batch_size: int    = 2,
    grad_accum: int    = 4,
    lr: float          = 2e-4,
    max_seq_len: int   = 1024,
    lora_r: int        = 16,
    lora_alpha: int    = 32,
    lora_dropout: float = 0.05,
):

    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer
    import torch

    print(f"[QLoRA] Loading base model: {base_model}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    )
    model = prepare_model_for_kbit_training(model)


    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    raw = [json.loads(l) for l in open(dataset_path, encoding="utf-8")]
    hf_dataset = Dataset.from_list([{"text": _format_alpaca(s)} for s in raw])
    split = hf_dataset.train_test_split(test_size=0.05, seed=42)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",           
        optim="paged_adamw_8bit",   
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
    )

    print("[QLoRA] Starting training…")
    trainer.train()

    print(f"[QLoRA] Saving LoRA adapter → {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[QLoRA] Done.")


def load_finetuned_model(adapter_path: str = ADAPTER_OUTPUT):

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for AI Tutor")
    parser.add_argument("--generate_data", action="store_true",
                        help="Generate training data from RAG corpus")
    parser.add_argument("--train", action="store_true",
                        help="Run QLoRA fine-tuning")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of training samples to generate")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()

    if args.generate_data:
        generate_dataset_from_corpus(n_samples=args.n_samples)

    if args.train:
        train_qlora(num_epochs=args.epochs, lora_r=args.lora_r)

    if not args.generate_data and not args.train:
        parser.print_help()
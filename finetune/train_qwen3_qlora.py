import os
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


TRAIN_PATH = "350-50/train.jsonl"
VAL_PATH   = "350-50/validation.jsonl"
OUTPUT_DIR = "qwen3_planner_recipe2_lora"
MODEL_ID = "Qwen/Qwen3-1.7B"
MAX_SEQ_LENGTH = 1664
SEED = 42

# Recipe2 defaults
LR = 1e-4
EPOCHS = 10
MICRO_BATCH_SIZE = 1
GRAD_ACCUM = 32
WARMUP_RATIO = 0.05


# -----------------------------
# Collator: pads input_ids/labels
# -----------------------------
@dataclass
class CausalLMCollator:
    """
    - pad input_ids with pad_token_id
    - pad attention_mask with 0
    - pad labels with -100
    - returns batch dict which model can consume.
    """
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            ids = f["input_ids"]
            attn = f["attention_mask"]
            lab = f["labels"]

            pad = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append(attn + [0] * pad)
            labels.append(lab + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class Config:
    output_dir: str = OUTPUT_DIR
    model_id: str = MODEL_ID
    max_seq_length: int = MAX_SEQ_LENGTH
    seed: int = SEED

    lr: float = LR
    epochs: int = EPOCHS
    micro_bsz: int = MICRO_BATCH_SIZE
    grad_accum: int = GRAD_ACCUM
    warmup_ratio: float = WARMUP_RATIO


def main():
    args = Config()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------
    # Tokenizer (Qwen3 chat template)
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
    )

    # Ensure pad token exists for batching; common practice for CausalLM
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # QLoRA 4-bit config
    # -----------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16, # Forward computations use FP16
    )

    # -----------------------------
    # Model load (4-bit base, train adapters)
    # -----------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    # makes checkpointing work with PEFT (enables input grads, etc.)
    model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()

    # -----------------------------
    # LoRA config (Recipe2): attention-only
    # -----------------------------
    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    
    datasets = load_dataset(
        "json",
        data_files={"train": TRAIN_PATH, "validation": VAL_PATH},
    )

    train_ds = datasets["train"]
    validation_ds = datasets["validation"]


    max_len = args.max_seq_length

    # -----------------------------
    # Preprocess: build input_ids/labels with prefix masking
    # Key point: keep completion intact; truncate from left of prompt if needed.
    # -----------------------------
    def preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        """
        takes a row of Dataset obj
        """
        messages = ex["messages"]

        prompt_messages = messages[:-1]
        assistant_text = messages[-1]["content"]

        # Force non-thinking mode in the formatted prefix
        # enable_thinking defaults to True, so set it explicitly to False.
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True, # appends the template’s “assistant starts here” marker, so the prompt ends right where the model would begin generating.
            enable_thinking=False,
        )

        # Tokenize prefix WITHOUT adding special tokens (template already includes them)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

        # Tokenize completion text
        completion_ids = tokenizer(assistant_text, add_special_tokens=False).input_ids
        eos_id = tokenizer.eos_token_id

        # print("PROMPT_TEXT (tail):", prompt_text[-400:])
        # print("ASSISTANT_TEXT (head):", assistant_text[:200])

        if len(completion_ids) == 0:raise RuntimeError("plan is empty!")

        # Ensure completion ends cleanly (optional but recommended)
        if eos_id is not None and  completion_ids[-1] != eos_id:
            completion_ids = completion_ids + [eos_id]

        # Truncate prefix from the LEFT to fit max_len, keeping completion intact - should happen for only on sample
        total_len = len(prompt_ids) + len(completion_ids)
        if total_len > max_len:
            print("MAX LEN EXCEEDED")
            overflow = total_len - max_len
            # Keep at least 1 prefix token (so completion is conditioned on something)
            keep = max(1, len(prompt_ids) - overflow)
            prompt_ids = prompt_ids[-keep:]

        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)

        # Mask prefix labels; supervise only completion tokens
        labels = [-100] * len(prompt_ids) + completion_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    validation_ds = validation_ds.map(preprocess, remove_columns=validation_ds.column_names)
    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)

    # Steps info (useful sanity check)
    steps_per_epoch = math.ceil(len(train_ds) / (args.micro_bsz * args.grad_accum))
    print(f"Train examples: {len(train_ds)}, Val examples: {len(validation_ds)}")
    print(f"Approx steps/epoch: {steps_per_epoch}")

    # -----------------------------
    # TrainingArguments (Recipe2)
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_bsz,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        max_grad_norm=1.0,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch", # at the end of each checkpoint, saves the adapters
        save_total_limit=5,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",  # good default for (Q)LoRA + bnb
        load_best_model_at_end=True, # saves the best model + 2 most recent ones due to `save_total_limit`
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save adapter + tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Best checkpoint:", trainer.state.best_model_checkpoint)

    print(f"Saved LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    main()
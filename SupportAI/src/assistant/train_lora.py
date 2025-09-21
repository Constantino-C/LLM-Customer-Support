import os, json, math, yaml, argparse
from typing import Optional
from datasets import load_dataset
from transformers import (
AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def main(config_path: str):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['save_dir'], exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    load_kwargs = {"trust_remote_code": True}
    if cfg['use_qlora']:
        load_kwargs.update(dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16" if cfg['bf16'] else "float16",
        ))

    model = AutoModelForCausalLM.from_pretrained(cfg['base_model'], **load_kwargs)
    if cfg['use_qlora']:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
    r=cfg['lora_r'],
    lora_alpha=cfg['lora_alpha'],
    lora_dropout=cfg['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    # Data
    ds_train = load_dataset("json", data_files=cfg['train_file'], split="train")
    ds_val = load_dataset("json", data_files=cfg['val_file'], split="train")

    def tok(batch):
        return tokenizer(
            batch["text"],
            max_length=cfg['max_seq_len'],
            truncation=True,
            padding=False,
        )

    ds_train = ds_train.map(tok, batched=True, remove_columns=["text"]).shuffle(seed=cfg['seed'])
    ds_val = ds_val.map(tok, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=cfg['save_dir'],
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['grad_accum'],
        learning_rate=cfg['lr'],
        max_steps=cfg['max_steps'],
        warmup_ratio=cfg['warmup_ratio'],
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=5,
        save_steps=10,
        save_total_limit=2,
        bf16=cfg['bf16'],
        report_to=["none"],
        seed=cfg['seed'],
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter only
    model.save_pretrained(cfg['save_dir'])
    tokenizer.save_pretrained(cfg['save_dir'])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", type=str ,default="configs/train.yaml")
    args = ap.parse_args()
    main(args.config_path)
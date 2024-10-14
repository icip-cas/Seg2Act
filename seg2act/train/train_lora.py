import inspect
import json
import os
from typing import List

import fire
import torch
import numpy as np
import random
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)
import pathlib
from seg2act.utils.utils import generate_prompt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(
    # model / data params
    data_path: str,
    output_dir: str,
    base_model: str = "", 
    # training hyperparams
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,   
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,    
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["W_pack"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
):
    setup_seed(seed)

    print(
        f"Training model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"seed: {seed}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
    )
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    sig = inspect.signature(train)
    param_names = [p.name for p in sig.parameters.values()]
    param_value = locals()
    params = {name: param_value[name] for name in param_names}

    os.makedirs(output_dir, exist_ok=True)
        
    with open(os.path.join(output_dir, "params.json"), "w") as fp:
        json.dump(params, fp=fp)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt, 
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
            
        result["labels"] = result["input_ids"].copy()
        return result
    
    def generate_and_tokenize_prompt(datapoint):
        full_prompt = generate_prompt(datapoint)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**datapoint, "action": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len \
                + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    model.print_trainable_parameters() # Be more transparent about the % of trainable params.
    data = load_dataset("json", data_files={"train": data_path}, num_proc=1)
    train_data = data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=TrainingArguments(
            seed=seed,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            save_strategy="epoch",
            output_dir=output_dir,
            group_by_length=group_by_length,
            save_total_limit=1,
            report_to="none",
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True,
        ),
    )
    
    if list(pathlib.Path(output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    
    state_dict = {k: t for k, t in model.named_parameters() if "lora_" in k}
    model.save_pretrained(output_dir, state_dict=state_dict)


if __name__ == "__main__":
    fire.Fire(train)
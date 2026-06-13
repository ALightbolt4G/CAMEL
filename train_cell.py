import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser(description="CAMEL LoRA Fine-Tuning")
    parser.add_argument("--cell", type=str, required=True, help="Cell name (e.g., code_cell, history_cell)")
    parser.add_argument("--base_model", type=str, default="/mnt/d/models/bloom-560m", help="Base model to fine-tune")
    args = parser.parse_args()
    
    cell_name = args.cell
    train_file = f"data/{cell_name}/train.jsonl"
    eval_file = f"data/{cell_name}/eval.jsonl"
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found at {train_file}. Run prepare.py first.")
        
    print(f"[*] Training {cell_name} using base model {args.base_model}...")
    
    # 1. Load Dataset
    dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Load Model in fp16
    print("Loading model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 4. LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"], # Target attention layers for BLOOM
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 5. Training Arguments
    training_args = SFTConfig(
        output_dir=f"cells/{cell_name}/checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        num_train_epochs=3,
        fp16=True,
        max_length=512,
        dataset_text_field="text"
    )
    
    # 6. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=training_args,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
    print(f"Starting Training for {cell_name}...")
    trainer.train()
    
    # 7. Save the Adapter
    adapter_path = f"cells/{cell_name}/adapter"
    os.makedirs(adapter_path, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[SUCCESS] Adapter saved successfully at {adapter_path}")

if __name__ == "__main__":
    main()

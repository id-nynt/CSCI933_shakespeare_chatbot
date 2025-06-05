# FINE-TUNE TINI-LLAMA + LORA

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

# === CONFIG ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
train_file = "data/fine_tuning/set1/dataset_tinyllama.jsonl"
output_dir = "models/tini_llama_01"
max_length = 512

# === Load dataset ===
dataset = load_dataset("json", data_files={"train": train_file}, split="train")

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Preprocessing function ===
def preprocess(example):
    full_text = example["text"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    # Causal LM label: same as input_ids
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids
    }

# === Tokenize dataset ===
tokenized_dataset = dataset.map(preprocess, remove_columns=["text"])
dataset_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# === Load base model ===
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

# === Setup LoRA config ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # specific to TinyLlama/LLaMA-style
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# === Training args ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_dir=f"{output_dir}/logs",
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    eval_strategy="epoch",
    fp16=torch.cuda.is_available(),
    bf16=False,
    report_to="none",
    remove_unused_columns=True
)

# === Data collator ===
data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# === Start training ===
if __name__ == "__main__":
    model.print_trainable_parameters()
    try:
        trainer.train()
        trainer.save_model(output_dir)
    except Exception as e:
        print(f"Training failed: {e}")
        trainer.save_model(output_dir)
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# === CONFIGURATION ===
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_model_dir = "models/tini_llama_01"  # Update model
test_file_path = "data/fine_tuning/set1/test.jsonl"  # Update test set
num_samples = 30
max_new_tokens = 100

# === LOAD TOKENIZER & MODEL ===
print("🔁 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, lora_model_dir)
model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# === LOAD TEST DATA ===
print(f"📄 Loading test samples from: {test_file_path}")
with open(test_file_path, "r", encoding="utf-8") as f:
    all_samples = [json.loads(line) for line in f]

if len(all_samples) < num_samples:
    print(f"⚠️ Not enough samples in dataset ({len(all_samples)} found), reducing sample size.")
    num_samples = len(all_samples)

test_samples = random.sample(all_samples, num_samples)

# === FORMAT HELPER ===
def build_chat_prompt(user_message):
    return (
        "<|system|>\n"
        "You are a helpful assistant knowledgeable in Shakespearean literature.\n"
        "<|user|>\n"
        f"{user_message}\n"
        "<|assistant|>\n"
    )

# === EVALUATION ===
print("\n🚀 === TINYLLAMA EVALUATION START ===\n")
for i, sample in enumerate(test_samples):
    full_text = sample.get("text", "")
    try:
        user_message = full_text.split("<|user|>\n")[1].split("\n<|assistant|>")[0].strip()
        reference = full_text.split("<|assistant|>\n")[1].strip()
    except IndexError:
        print(f"❌ Skipping invalid format at sample {i}")
        continue

    prompt = build_chat_prompt(user_message)

    print(f"🔹 Q{i+1}: {user_message}")
    print(f"📘 Expected Answer: {reference}")

    try:
        output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0]["generated_text"].replace(prompt, "").strip()
    except Exception as e:
        generated = f"❌ Error generating: {e}"

    print(f"🤖 Model Answer: {generated}")
    print("-" * 100)
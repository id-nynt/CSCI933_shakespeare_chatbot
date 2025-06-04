# Process dataset to instruction format for Tini Llama
import json

input_path = "data/fine_tuning/set1/combined_prompt.jsonl"
output_path = "data/fine_tuning/set1/dataset_tinyllama.jsonl"

def format_for_tinyllama(example):
    return {
        "text": f"<|system|>\nYou are a helpful assistant knowledgeable in Shakespearean literature.\n"
                f"<|user|>\n{example['prompt']}\n"
                f"<|assistant|>\n{example['response']}"
    }

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            sample = json.loads(line.strip())
            formatted = format_for_tinyllama(sample)
            outfile.write(json.dumps(formatted, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Skipping a line due to error: {e}")

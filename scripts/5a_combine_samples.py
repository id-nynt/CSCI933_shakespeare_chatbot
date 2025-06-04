import os
import json
import logging
from collections import defaultdict

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = "data/prompt_response/set1"
OUTPUT_DIR = "data/fine_tuning/set1"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset_tinyllama.jsonl")

# === HELPERS ===
def is_valid_jsonl_line(line, file_path):
    try:
        data = json.loads(line.strip())
        if not isinstance(data, dict) or "prompt" not in data or "response" not in data:
            logger.warning(f"Invalid JSONL line in {file_path}: Missing prompt or response")
            return False
        return True
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in {file_path}: {line.strip()}")
        return False

def format_pair(data):
    prompt = data["prompt"].strip()
    response = data["response"].strip().rstrip("?")

    return {
        "prompt": prompt,
        "response": response,
        "question_type": data.get("question_type", "unknown"),
        "play": data.get("play", "unknown")
    }

def combine_jsonl_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input directory {INPUT_DIR} does not exist")
        return 0

    jsonl_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    if not jsonl_files:
        logger.error(f"No JSONL files found in {INPUT_DIR}")
        return 0

    total_pairs = 0
    question_type_counts = defaultdict(int)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for filename in jsonl_files:
            file_path = os.path.join(INPUT_DIR, filename)
            logger.info(f"Processing file: {file_path}")
            file_pairs = 0

            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        if line.strip() and is_valid_jsonl_line(line, file_path):
                            data = json.loads(line.strip())
                            formatted = format_pair(data)
                            outfile.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                            total_pairs += 1
                            file_pairs += 1
                            question_type_counts[formatted["question_type"]] += 1
                logger.info(f"Added {file_pairs} pairs from {filename}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"✅ Combined {total_pairs} prompt-response pairs into {OUTPUT_FILE}")
    print(f"\n📊 Total prompt-response pairs: {total_pairs}")
    print("📂 Breakdown by question_type:")
    for qtype, count in sorted(question_type_counts.items(), key=lambda x: x[0]):
        print(f"  {qtype}: {count}")

    return total_pairs

# === RUN ===
if __name__ == "__main__":
    combine_jsonl_files()
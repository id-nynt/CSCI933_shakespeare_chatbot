import os
import json
import random
from pathlib import Path

# Define input and output paths
input_dir = Path("data/processed/dialogue")
output_file = Path("data/prompt_response/dialogue_qa.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

# List of target plays
plays = [
    "hamlet", "romeo_and_juliet", "macbeth", "midsummer_nights_dream", "othello",
    # "much_ado_about_nothing", "king_lear", "twelfth_night",
    # "the_tempest", "julius_caesar",
]

# Updated question paraphrases with descriptive question type labels
question_types = {
    "next_quote": [
        "What is the quote after '{line}' in '{title}'?",
        "What comes next after the line '{line}' in '{title}'?",
        "Which quote follows this in '{title}': '{line}'?",
        "Tell me the next line after '{line}' in the play '{title}'?",
        "Who speaks next and what do they say after '{line}' in '{title}'?"
    ],
    "scene_characters": [
        "Who are the characters in scene {scene}, act {act} of '{title}'?",
        "List characters in Act {act}, Scene {scene} of '{title}'",
        "Which characters appear in Act {act}, Scene {scene} of the play '{title}'?",
        "Who is present in Act {act}, Scene {scene} of '{title}'?",
        "Can you name the characters in Act {act}, Scene {scene} from '{title}'?"
    ],
    "speaker": [
        "Who said the line '{line}' in '{title}'?",
        "Identify the speaker of '{line}' in the play '{title}'",
        "Which character said: '{line}' in '{title}'?",
        "Find the speaker of the quote '{line}' from '{title}'",
        "Who spoke this line in '{title}': '{line}'?"
    ],
    "play_source": [
        "Which play contains the quote: '{line}'?",
        "In which Shakespeare play does the line '{line}' appear?",
        "What play has this line: '{line}'?",
        "Find the play that includes the line: '{line}'",
        "Which work includes the following quote: '{line}'?"
    ],
    "scene_location": [
        "Where does Act {act}, Scene {scene} of '{title}' take place?",
        "What is the setting of Act {act}, Scene {scene} in '{title}'?",
        "Can you describe the location of Act {act}, Scene {scene} in '{title}'?",
        "In what place is Act {act}, Scene {scene} set in '{title}'?",
        "What is the location for Act {act}, Scene {scene} of the play '{title}'?"
    ]
}

samples = []
max_samples_per_play = 10000  # Limit for each question type per play
collected = {play: {"next_quote": 0, "scene_characters": 0, "speaker": 0, "play_source": 0, "scene_location": 0} for play in plays}

for play in plays:
    path = input_dir / f"{play}.json"
    if not path.exists():
        print(f"Warning: {path} not found, skipping {play}")
        continue

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {path}: {e}")
            continue

    title = data.get("title", play.replace("_", " ").title())

    for act_obj in data.get("acts", []):
        act = act_obj.get("act")
        for scene_obj in act_obj.get("scenes", []):
            scene = scene_obj.get("scene")
            dialogues = scene_obj.get("dialogues", [])
            characters = scene_obj.get("scene_characters", [])
            location = scene_obj.get("location")

            # Question: Scene characters (up to max_samples_per_play)
            if collected[play]["scene_characters"] < max_samples_per_play:
                answer = ", ".join(characters) if characters else "No characters specified"
                # Generate two different question paraphrases for the same answer
                selected_questions = random.sample(question_types["scene_characters"], min(2, len(question_types["scene_characters"])))
                for question_template in selected_questions:
                    question = question_template.format(title=title, act=act, scene=scene)
                    samples.append({"prompt": question, "response": answer, "question_type": "scene_characters", "play": title})
                    collected[play]["scene_characters"] += 1

            # Question: Scene location (up to max_samples_per_play)
            if location and collected[play]["scene_location"] < max_samples_per_play:
                answer = location
                # Generate two different question paraphrases for the same answer
                selected_questions = random.sample(question_types["scene_location"], min(2, len(question_types["scene_location"])))
                for question_template in selected_questions:
                    question = question_template.format(title=title, act=act, scene=scene)
                    samples.append({"prompt": question, "response": answer, "question_type": "scene_location", "play": title})
                    collected[play]["scene_location"] += 1

            for i, dialogue in enumerate(dialogues):
                speaker = dialogue.get("speaker")
                line = dialogue.get("line")

                if not line or not speaker:
                    continue

                # Question: Next quote (up to max_samples_per_play)
                if i + 1 < len(dialogues) and collected[play]["next_quote"] < max_samples_per_play:
                    next_dialogue = dialogues[i + 1]
                    answer = f"{next_dialogue['speaker']}: {next_dialogue['line']}"
                    # Generate two different question paraphrases for the same answer
                    selected_questions = random.sample(question_types["next_quote"], min(2, len(question_types["next_quote"])))
                    for question_template in selected_questions:
                        question = question_template.format(line=line, title=title)
                        samples.append({"prompt": question, "response": answer, "question_type": "next_quote", "play": title})
                        collected[play]["next_quote"] += 1

                # Question: Speaker of the line (up to max_samples_per_play)
                if collected[play]["speaker"] < max_samples_per_play:
                    answer = f"{speaker} in Act {act}, Scene {scene}"
                    # Generate two different question paraphrases for the same answer
                    selected_questions = random.sample(question_types["speaker"], min(2, len(question_types["speaker"])))
                    for question_template in selected_questions:
                        question = question_template.format(line=line, title=title)
                        samples.append({"prompt": question, "response": answer, "question_type": "speaker", "play": title})
                        collected[play]["speaker"] += 1

                # Question: Play containing the line (up to max_samples_per_play)
                if collected[play]["play_source"] < max_samples_per_play:
                    answer = title
                    # Generate two different question paraphrases for the same answer
                    selected_questions = random.sample(question_types["play_source"], min(2, len(question_types["play_source"])))
                    for question_template in selected_questions:
                        question = question_template.format(line=line)
                        samples.append({"prompt": question, "response": answer, "question_type": "play_source", "play": title})
                        collected[play]["play_source"] += 1

# Save output as JSONL
with open(output_file, 'w', encoding='utf-8') as f:
    for sample in samples:
        json.dump(sample, f, ensure_ascii=False)
        f.write('\n')

# Print summary
total_samples = len(samples)
print(f"Saved {total_samples} samples to {output_file}")
print("Collected per play and type:")
for play in plays:
    if play in collected:
        print(f"{play}: {collected[play]}")
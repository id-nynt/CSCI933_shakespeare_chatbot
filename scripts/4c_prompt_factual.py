import os
import json
import random
import logging

# === CONFIG ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = "data/processed/factual/factual.json"
OUTPUT_FILE = "data/prompt_response/factual_qa.jsonl"
PLAYS = [
    "Hamlet, Prince of Denmark",
    "Romeo and Juliet",
    "Macbeth",
    "Othello, the Moor of Venice",
    "A Midsummer Night's Dream",
    "Much Ado About Nothing",
    "The Tragedy of King Lear",
    "Twelfth Night; Or, What You Will",
    "The Tempest",
    "Julius Caesar",
]

# === QUESTION TEMPLATES ===
QUESTION_TEMPLATES = {
    "theme_specific": [
        lambda t, th: f"How is the theme of {th} explored in *{t}*?",
        lambda t, th: f"Explain the theme of {th} in *{t}*?",
        lambda t, th: f"Can you explain how *{t}* addresses the theme of {th}?"
    ],
    "category_list": [
        lambda c: f"Which plays are in the {c} category?",
        lambda c: f"Can you name all the plays classified as {c}?",
        lambda c: f"What are the titles of Shakespeare’s {c} plays?"
    ],
    "main_characters": [
        lambda t: f"Who are the primary characters in *{t}*?",
        lambda t: f"Can you list the key characters in *{t}*?",
        lambda t: f"Who are the main characters in *{t}*?"
    ],
    "character_role": [
        lambda t, n: f"What is the role of {n} in *{t}*?",
        lambda t, n: f"Describe {n}’s character in *{t}*?",
        lambda t, n: f"Who is {n} in *{t}*?"
    ],
    "themes_general": [
        lambda t: f"Which themes are central to *{t}*?",
        lambda t: f"What are the main themes in *{t}*?",
        lambda t: f"What are the key thematic elements in *{t}*?"
    ],
    "year_written": [
        lambda t: f"When was *{t}* written?",
        lambda t: f"What is the composition year of *{t}*?",
        lambda t: f"When did Shakespeare write *{t}*?"
    ],
    "setting": [
        lambda t: f"What is the setting of *{t}*?",
        lambda t: f"Where is *{t}* set?",
        lambda t: f"In what location does *{t}* occur?"
    ],
    "category": [
        lambda t: f"Which category is *{t}* in?",
        lambda t: f"What type of play is *{t}* classified as?",
        lambda t: f"Can you tell me the genre of *{t}*?"
    ]
}

# === HELPERS ===
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return None

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# === MAIN FUNCTION ===
def generate_factual_qa():
    qa_dataset = []
    data = load_json(INPUT_FILE)
    if not data:
        logger.error("No data loaded, exiting")
        return

    # Filter for requested plays
    plays_data = [play for play in data if play.get("title") in PLAYS]
    missing_plays = [title for title in PLAYS if title not in [p.get("title") for p in plays_data]]
    if missing_plays:
        logger.warning(f"Missing plays in factual.json: {missing_plays}")

    # Collect unique categories for question 2
    categories = set(play.get("category", "").lower() for play in plays_data if play.get("category"))
    category_plays = {cat: [p["title"] for p in plays_data if p.get("category", "").lower() == cat] for cat in categories}

    for play in plays_data:
        title = play.get("title", "")
        if not title:
            logger.warning("Skipping play with no title")
            continue

        logger.info(f"Processing play: {title}")

        # Question 1: Theme-specific questions
        for theme_entry in play.get("themes", []):
            theme = theme_entry.get("theme", "")
            explanation = theme_entry.get("theme_explanation", "")
            if theme and explanation:
                # Select 2 unique question templates
                selected_templates = random.sample(QUESTION_TEMPLATES["theme_specific"], min(2, len(QUESTION_TEMPLATES["theme_specific"])))
                for template in selected_templates:
                    question = template(title, theme)
                    qa_dataset.append({"prompt": question, "response": explanation, "question_type": "theme_specific", "play": title})

        # Question 2: Category list (handled later for all categories)

        # Question 3: Main characters
        main_characters = play.get("main_characters", [])
        if main_characters:
            # Select 2 unique question templates
            selected_templates = random.sample(QUESTION_TEMPLATES["main_characters"], min(2, len(QUESTION_TEMPLATES["main_characters"])))
            response = ", ".join(main_characters) + "."
            for template in selected_templates:
                question = template(title)
                qa_dataset.append({"prompt": question, "response": response, "question_type": "main_characters", "play": title})

        # Question 4: Character roles
        for char_entry in play.get("character_descriptions", []):
            name = char_entry.get("name", "")
            description = char_entry.get("description", "")
            if name and description:
                # Select 2 unique question templates
                selected_templates = random.sample(QUESTION_TEMPLATES["character_role"], min(2, len(QUESTION_TEMPLATES["character_role"])))
                for template in selected_templates:
                    question = template(title, name)
                    qa_dataset.append({"prompt": question, "response": description, "question_type": "character_role", "play": title})

        # Question 5: General themes
        themes = [t.get("theme", "") for t in play.get("themes", []) if t.get("theme")]
        if themes:
            # Select 2 unique question templates
            selected_templates = random.sample(QUESTION_TEMPLATES["themes_general"], min(2, len(QUESTION_TEMPLATES["themes_general"])))
            response = ", ".join(themes) + "."
            for template in selected_templates:
                question = template(title)
                qa_dataset.append({"prompt": question, "response": response, "question_type": "themes_general", "play": title})

        # Question 6: Year written
        year = play.get("year", "")
        if year:
            # Select 2 unique question templates
            selected_templates = random.sample(QUESTION_TEMPLATES["year_written"], min(2, len(QUESTION_TEMPLATES["year_written"])))
            for template in selected_templates:
                question = template(title)
                qa_dataset.append({"prompt": question, "response": str(year), "question_type": "year_written", "play": title})

        # Question 7: Setting
        setting = play.get("setting", "")
        if setting:
            # Select 2 unique question templates
            selected_templates = random.sample(QUESTION_TEMPLATES["setting"], min(2, len(QUESTION_TEMPLATES["setting"])))
            for template in selected_templates:
                question = template(title)
                qa_dataset.append({"prompt": question, "response": setting, "question_type": "setting", "play": title})

        # Question 8: Category
        category = play.get("category", "")
        if category:
            # Select 2 unique question templates
            selected_templates = random.sample(QUESTION_TEMPLATES["category"], min(2, len(QUESTION_TEMPLATES["category"])))
            for template in selected_templates:
                question = template(title)
                qa_dataset.append({"prompt": question, "response": category.capitalize(), "question_type": "category", "play": title})

    # Question 2: List plays by category
    for category in categories:
        plays_in_category = category_plays.get(category, [])
        if plays_in_category:
            # Select 2 unique question templates
            selected_templates = random.sample(QUESTION_TEMPLATES["category_list"], min(2, len(QUESTION_TEMPLATES["category_list"])))
            response = ", ".join(plays_in_category) + "."
            for template in selected_templates:
                question = template(category.capitalize())
                qa_dataset.append({"prompt": question, "response": response, "question_type": "category_list"})

    save_jsonl(qa_dataset, OUTPUT_FILE)
    logger.info(f"✅ Generated {len(qa_dataset)} prompt-response pairs and saved to {OUTPUT_FILE}")

# === RUN ===
if __name__ == "__main__":
    generate_factual_qa()
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str
    play_title: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float] = None

class EnhancedShakespeareChunker:
    def __init__(self, data_dir: str = "data", output_dir: str = "retrieval/chunks"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.glossary = self.load_glossary()
        self.chunks = []
        
    def load_glossary(self) -> Dict[str, str]:
        """Load the Shakespeare glossary for reference."""
        glossary_path = self.data_dir / "glossary" / "glossary.json"
        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Glossary not found at {glossary_path}")
            return {}
    
    def generate_chunk_id(self, content: str, chunk_type: str, play_title: str) -> str:
        """Generate a unique chunk ID based on content hash."""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        safe_title = play_title.lower().replace(' ', '_').replace(',', '').replace('.', '')
        return f"{chunk_type}_{safe_title}_{content_hash}"
    
    def create_character_identity_chunks(self):
        """Create specific chunks for 'Who is X?' questions."""
        factual_path = self.data_dir / "processed" / "factual" / "factual.json"
        
        if not factual_path.exists():
            return
            
        try:
            with open(factual_path, 'r', encoding='utf-8') as f:
                factual_data = json.load(f)
            
            for play in factual_data:
                play_title = play.get("title", "Unknown Play")
                
                # Create individual character identity chunks
                for char_desc in play.get("character_descriptions", []):
                    name = char_desc.get("name", "Unknown")
                    description = char_desc.get("description", "No description")
                    
                    # Format content for "Who is X?" queries
                    char_content = f"{name} is {description}"
                    
                    chunk_id = self.generate_chunk_id(char_content, "character_identity", play_title)
                    
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        chunk_type="character_identity",
                        play_title=play_title,
                        content=char_content,
                        metadata={
                            "character_name": name,
                            "query_patterns": [f"who is {name.lower()}", f"who is {name.lower()} in {play_title.lower()}"]
                        }
                    )
                    
                    self.chunks.append(chunk)
                    
        except Exception as e:
            print(f"Error creating character identity chunks: {e}")
    
    def create_play_metadata_chunks(self):
        """Create chunks for factual questions about plays."""
        factual_path = self.data_dir / "processed" / "factual" / "factual.json"
        
        if not factual_path.exists():
            return
            
        try:
            with open(factual_path, 'r', encoding='utf-8') as f:
                factual_data = json.load(f)
            
            for play in factual_data:
                play_title = play.get("title", "Unknown Play")
                category = play.get("category", "Unknown")
                year = play.get("year", "Unknown")
                setting = play.get("setting", "Unknown")
                
                # When was the play written?
                year_content = f"{play_title} was written in {year}."
                
                year_chunk = Chunk(
                    chunk_id=self.generate_chunk_id(year_content, "play_year", play_title),
                    chunk_type="play_year",
                    play_title=play_title,
                    content=year_content,
                    metadata={"year": year, "query_patterns": [f"when was {play_title.lower()} written"]}
                )
                self.chunks.append(year_chunk)
                
                # Where does the play occur?
                setting_content = f"{play_title} takes place in {setting}."
                
                setting_chunk = Chunk(
                    chunk_id=self.generate_chunk_id(setting_content, "play_setting", play_title),
                    chunk_type="play_setting",
                    play_title=play_title,
                    content=setting_content,
                    metadata={"setting": setting, "query_patterns": [f"where does {play_title.lower()} take place", f"where was {play_title.lower()}"]}
                )
                self.chunks.append(setting_chunk)
                
                # What category is the play?
                category_content = f"The category of {play_title} is {category}."
                
                category_chunk = Chunk(
                    chunk_id=self.generate_chunk_id(category_content, "play_category", play_title),
                    chunk_type="play_category",
                    play_title=play_title,
                    content=category_content,
                    metadata={"category": category, "query_patterns": [f"what category is {play_title.lower()}", f"what type of play is {play_title.lower()}"]}
                )
                self.chunks.append(category_chunk)
                
                # Main characters list
                main_chars = play.get("main_characters", [])
                chars_content = f"The main characters in {play_title} are: {', '.join(main_chars)}."
                
                chars_chunk = Chunk(
                    chunk_id=self.generate_chunk_id(chars_content, "main_characters", play_title),
                    chunk_type="main_characters",
                    play_title=play_title,
                    content=chars_content,
                    metadata={"main_characters": main_chars, "query_patterns": [f"main characters in {play_title.lower()}", f"who are the characters in {play_title.lower()}"]}
                )
                self.chunks.append(chars_chunk)
                    
        except Exception as e:
            print(f"Error creating play metadata chunks: {e}")
    
    def create_dialogue_context_chunks(self):
        """Create chunks for multi-turn dialogue questions."""
        dialogue_dir = self.data_dir / "processed" / "dialogue"
        
        if not dialogue_dir.exists():
            return
            
        for file_path in dialogue_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    play_data = json.load(f)
                
                play_title = play_data.get("title", "Unknown Play")
                
                for act in play_data.get("acts", []):
                    act_num = act.get("act", 0)
                    
                    for scene in act.get("scenes", []):
                        scene_num = scene.get("scene", 0)
                        dialogues = scene.get("dialogues", [])
                        
                        # Create chunks for dialogue sequences
                        for i, dialogue in enumerate(dialogues):
                            speaker = dialogue.get("speaker", "Unknown")
                            line = dialogue.get("line", "")
                            
                            # Create "who said this line" chunk
                            line_speaker_content = f"In {play_title}, {speaker} said the line: \"{line}\""
                            
                            line_chunk = Chunk(
                                chunk_id=self.generate_chunk_id(line_speaker_content, "line_speaker", play_title),
                                chunk_type="line_speaker",
                                play_title=play_title,
                                content=line_speaker_content,
                                metadata={
                                    "speaker": speaker,
                                    "line": line,
                                    "act": act_num,
                                    "scene": scene_num,
                                    "line_index": i
                                }
                            )
                            self.chunks.append(line_chunk)
                            
                            # Create "who speaks next" chunk
                            if i < len(dialogues) - 1:
                                next_dialogue = dialogues[i + 1]
                                next_speaker = next_dialogue.get("speaker", "Unknown")
                                next_line = next_dialogue.get("line", "")
                                
                                next_speaker_content = f"In {play_title} after {speaker} said \"{line}\", {next_speaker} speaks next and says: \"{next_line}\""
                                
                                next_chunk = Chunk(
                                    chunk_id=self.generate_chunk_id(next_speaker_content, "next_speaker", play_title),
                                    chunk_type="next_speaker",
                                    play_title=play_title,
                                    content=next_speaker_content,
                                    metadata={
                                        "current_speaker": speaker,
                                        "current_line": line,
                                        "next_speaker": next_speaker,
                                        "next_line": next_line,
                                        "act": act_num,
                                        "scene": scene_num
                                    }
                                )
                                self.chunks.append(next_chunk)
                        
            except Exception as e:
                print(f"Error processing dialogue file {file_path}: {e}")
    
    def create_quote_lookup_chunks(self):
        """Create chunks optimized for quote-related questions."""
        quotes_path = self.data_dir / "processed" / "quote" / "quote.json"
        
        if not quotes_path.exists():
            return
            
        try:
            with open(quotes_path, 'r', encoding='utf-8') as f:
                quotes_data = json.load(f)
            
            for play in quotes_data:
                play_title = play.get("title", "Unknown Play")
                
                # Create a "list of quotes" chunk for the play
                all_quotes = []
                quote_contents = []
                
                for quote_data in play.get("famous_quotes", []):
                    quote = quote_data.get("quote", "")
                    speaker = quote_data.get("speaker", "Unknown")
                    act = quote_data.get("act", 0)
                    scene = quote_data.get("scene", 0)
                    explanation = quote_data.get("explanation", "No explanation")
                    
                    all_quotes.append(f'"{quote}" - {speaker}')
                    
                    # Individual quote meaning chunk
                    quote_meaning_content = f"The quote \"{quote}\" of {speaker} (Act {act}, Scene {scene}) means: {explanation}"
                    
                    quote_chunk = Chunk(
                        chunk_id=self.generate_chunk_id(quote_meaning_content, "quote_meaning", play_title),
                        chunk_type="quote_meaning",
                        play_title=play_title,
                        content=quote_meaning_content,
                        metadata={
                            "quote": quote,
                            "speaker": speaker,
                            "act": act,
                            "scene": scene,
                            "explanation": explanation
                        }
                    )
                    self.chunks.append(quote_chunk)
                    
                    # "Which play has this quote" chunk
                    play_quote_content = f"This quote \"{quote}\" is from {play_title}, spoken by {speaker}."
                    
                    play_quote_chunk = Chunk(
                        chunk_id=self.generate_chunk_id(play_quote_content, "quote_to_play", play_title),
                        chunk_type="quote_to_play",
                        play_title=play_title,
                        content=play_quote_content,
                        metadata={
                            "quote": quote,
                            "speaker": speaker
                        }
                    )
                    self.chunks.append(play_quote_chunk)
                
                # List all quotes chunk
                quotes_list_content = f"Famous quotes from {play_title}:\n"
                quotes_list_content += "\n".join([f"- {q}" for q in all_quotes])
                
                quotes_list_chunk = Chunk(
                    chunk_id=self.generate_chunk_id(quotes_list_content, "quotes_list", play_title),
                    chunk_type="quotes_list",
                    play_title=play_title,
                    content=quotes_list_content,
                    metadata={
                        "quote_count": len(all_quotes),
                        "query_patterns": [f"quotes from {play_title.lower()}", f"famous quotes {play_title.lower()}"]
                    }
                )
                self.chunks.append(quotes_list_chunk)
                    
        except Exception as e:
            print(f"Error processing quotes: {e}")
    
    def create_theme_chunks(self):
        """Create enhanced theme chunks for thematic questions."""
        factual_path = self.data_dir / "processed" / "factual" / "factual.json"
        
        if not factual_path.exists():
            return
            
        try:
            with open(factual_path, 'r', encoding='utf-8') as f:
                factual_data = json.load(f)
            
            for play in factual_data:
                play_title = play.get("title", "Unknown Play")
                themes = play.get("themes", [])
                
                # "What are the main themes" chunk
                theme_names = [theme.get("theme", "Unknown") for theme in themes]
                themes_list_content = f"The main themes in {play_title} are: {', '.join(theme_names)}."
                
                themes_list_chunk = Chunk(
                    chunk_id=self.generate_chunk_id(themes_list_content, "themes_list", play_title),
                    chunk_type="themes_list",
                    play_title=play_title,
                    content=themes_list_content,
                    metadata={
                        "themes": theme_names,
                        "query_patterns": [f"themes in {play_title.lower()}", f"main themes {play_title.lower()}"]
                    }
                )
                self.chunks.append(themes_list_chunk)
                
                # Individual theme exploration chunks
                for theme_data in themes:
                    theme_name = theme_data.get("theme", "Unknown Theme")
                    theme_explanation = theme_data.get("theme_explanation", "No explanation")
                    
                    theme_exploration_content = f"The theme of {theme_name} is explored in {play_title}: {theme_explanation}"
                    
                    theme_chunk = Chunk(
                        chunk_id=self.generate_chunk_id(theme_exploration_content, "theme_exploration", play_title),
                        chunk_type="theme_exploration",
                        play_title=play_title,
                        content=theme_exploration_content,
                        metadata={
                            "theme_name": theme_name,
                            "query_patterns": [f"theme of {theme_name.lower()} in {play_title.lower()}", f"how is {theme_name.lower()} explored"]
                        }
                    )
                    self.chunks.append(theme_chunk)
                    
        except Exception as e:
            print(f"Error creating theme chunks: {e}")
    
    def create_summary_chunks(self):
        """Create summary chunks for different levels."""
        summary_dir = self.data_dir / "processed" / "full_summary"
        
        if not summary_dir.exists():
            return
            
        for file_path in summary_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                play_title = summary_data.get("title", "Unknown Play")
                
                # Play-level summary
                play_summary = summary_data.get('play_summary', 'No summary available')
                play_summary_content = f"Summary of {play_title}: {play_summary}"
                
                play_chunk = Chunk(
                    chunk_id=self.generate_chunk_id(play_summary_content, "play_summary", play_title),
                    chunk_type="play_summary",
                    play_title=play_title,
                    content=play_summary_content,
                    metadata={
                        "summary_level": "play",
                        "query_patterns": [f"summarize {play_title.lower()}", f"summary of {play_title.lower()}"]
                    }
                )
                self.chunks.append(play_chunk)
                
                # Act-level summaries
                for act in summary_data.get("acts", []):
                    act_num = act.get("act", 0)
                    act_summary = act.get("act_summary", "No summary available")
                    
                    act_content = f"Summary of Act {act_num} from {play_title}: {act_summary}"
                    
                    act_chunk = Chunk(
                        chunk_id=self.generate_chunk_id(act_content, "act_summary", play_title),
                        chunk_type="act_summary",
                        play_title=play_title,
                        content=act_content,
                        metadata={
                            "summary_level": "act",
                            "act": act_num,
                            "query_patterns": [f"summarize act {act_num} {play_title.lower()}", f"act {act_num} summary {play_title.lower()}"]
                        }
                    )
                    self.chunks.append(act_chunk)
                    
                    # Scene-level summaries
                    for scene in act.get("scenes", []):
                        scene_num = scene.get("scene", 0)
                        location = scene.get("location", "Unknown location")
                        scene_summary = scene.get("scene_summary", "No summary available")
                        
                        scene_content = f"Summary of Act {act_num}, Scene {scene_num} from {play_title}: {scene_summary}"
                        
                        scene_chunk = Chunk(
                            chunk_id=self.generate_chunk_id(scene_content, "scene_summary", play_title),
                            chunk_type="scene_summary",
                            play_title=play_title,
                            content=scene_content,
                            metadata={
                                "summary_level": "scene",
                                "act": act_num,
                                "scene": scene_num,
                                "location": location,
                                "query_patterns": [f"summarize act {act_num} scene {scene_num} {play_title.lower()}"]
                            }
                        )
                        self.chunks.append(scene_chunk)
                        
            except Exception as e:
                print(f"Error processing summary file {file_path}: {e}")
    
    def create_character_relationship_chunks(self):
        """Create enhanced character relationship chunks."""
        relationships_path = self.data_dir / "glossary" / "character_relationship.json"
        
        if not relationships_path.exists():
            return
            
        try:
            with open(relationships_path, 'r', encoding='utf-8') as f:
                relationships_data = json.load(f)
            
            for play_data in relationships_data:
                play_title = play_data.get("title", "Unknown Play")
                
                for rel_data in play_data.get("character_descriptions", []):
                    char1 = rel_data.get("character_1", "Unknown")
                    char2 = rel_data.get("character_2", "Unknown")
                    rel_type = rel_data.get("relationship_type", "Unknown")
                    description = rel_data.get("description", "No description")
                    
                    rel_content = f"Relationship of {char1} and {char2} in {play_title} are {rel_type}. {description}"
                    
                    rel_chunk = Chunk(
                        chunk_id=self.generate_chunk_id(rel_content, "character_relationship", play_title),
                        chunk_type="character_relationship",
                        play_title=play_title,
                        content=rel_content,
                        metadata={
                            "character_1": char1,
                            "character_2": char2,
                            "relationship_type": rel_type,
                            "query_patterns": [f"{char1.lower()} {char2.lower()}", f"relationship {char1.lower()} {char2.lower()}"]
                        }
                    )
                    self.chunks.append(rel_chunk)
                    
        except Exception as e:
            print(f"Error processing character relationships: {e}")
    
    def save_chunks(self):
        """Save processed chunks to JSON files."""
        chunks_data = []
        for chunk in self.chunks:
            chunk_dict = asdict(chunk)
            chunks_data.append(chunk_dict)
        
        all_chunks_path = self.output_dir / "all_chunks.json"
        with open(all_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save chunks by type
        chunks_by_type = {}
        for chunk in self.chunks:
            chunk_type = chunk.chunk_type
            if chunk_type not in chunks_by_type:
                chunks_by_type[chunk_type] = []
            chunks_by_type[chunk_type].append(asdict(chunk))
        
        for chunk_type, type_chunks in chunks_by_type.items():
            type_path = self.output_dir / f"chunks_{chunk_type}.json"
            with open(type_path, 'w', encoding='utf-8') as f:
                json.dump(type_chunks, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            "total_chunks": len(self.chunks),
            "chunks_by_type": {k: len(v) for k, v in chunks_by_type.items()},
            "processed_timestamp": datetime.now().isoformat(),
            "chunk_types": list(chunks_by_type.keys())
        }
        
        metadata_path = self.output_dir / "chunks_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    def process_all(self):
        """Process all data sources and create enhanced chunks."""
        print("Processing Shakespeare data into enhanced chunks...")
        
        print("1. Creating character identity chunks...")
        self.create_character_identity_chunks()
        
        print("2. Creating play metadata chunks...")
        self.create_play_metadata_chunks()
        
        print("3. Creating dialogue context chunks...")
        self.create_dialogue_context_chunks()
        
        print("4. Creating quote lookup chunks...")
        self.create_quote_lookup_chunks()
        
        print("5. Creating theme chunks...")
        self.create_theme_chunks()
        
        print("6. Creating summary chunks...")
        self.create_summary_chunks()
        
        print("7. Creating character relationship chunks...")
        self.create_character_relationship_chunks()
        
        print("8. Saving chunks...")
        metadata = self.save_chunks()
        
        print("\nProcessing complete!")
        print(f"Total chunks created: {metadata['total_chunks']}")
        print("Chunks by type:")
        for chunk_type, count in metadata['chunks_by_type'].items():
            print(f"  - {chunk_type}: {count}")
        print(f"\nChunks saved to: {self.output_dir}")
        
        return metadata

def main():
    """Main function to run the enhanced chunking process."""
    chunker = EnhancedShakespeareChunker()
    metadata = chunker.process_all()
    
    # Display example chunks
    print("\n" + "="*50)
    print("ENHANCED CHUNK EXAMPLES:")
    print("="*50)
    
    chunk_types_to_show = ["character_identity", "quote_meaning", "themes_list", "line_speaker"]
    
    for chunk_type in chunk_types_to_show:
        matching_chunks = [chunk for chunk in chunker.chunks if chunk.chunk_type == chunk_type]
        if matching_chunks:
            example_chunk = matching_chunks[0]
            print(f"\n--- {chunk_type.upper()} CHUNK EXAMPLE ---")
            print(f"ID: {example_chunk.chunk_id}")
            print(f"Play: {example_chunk.play_title}")
            print(f"Content: {example_chunk.content[:300]}...")
            print(f"Metadata: {example_chunk.metadata}")

if __name__ == "__main__":
    main()
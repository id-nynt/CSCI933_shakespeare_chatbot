import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import difflib

@dataclass
class SearchResult:
    chunk_id: str
    chunk_type: str
    play_title: str
    content: str
    metadata: Dict[str, Any]
    distance: float
    relevance_score: float
    match_reasons: List[str]

class ShakespeareRAGSystem:
    def __init__(self, index_dir: str = "retrieval/index"):
        self.index_dir = Path(index_dir)
        self.model = None
        self.index = None
        self.chunks = None
        self.chunk_mapping = None
        self.metadata = None
        
        # Create lookup dictionaries for fast filtering
        self.chunks_by_type = defaultdict(list)
        self.chunks_by_play = defaultdict(list)
        
        self.load_index()
    
    def load_index(self):
        """Load the FAISS index and related data."""
        print("Loading RAG system...")

        # Set default model name
        default_model_name = 'all-MiniLM-L6-v2'
        model_name = default_model_name

        # Load metadata if exists
        metadata_path = self.index_dir / "index_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                model_name = self.metadata.get('model_name') or default_model_name
        else:
            self.metadata = {}

        # Safety check for invalid model name
        if model_name.strip().lower() == "unknown":
            print(f"⚠️  Invalid model name in metadata. Using default: {default_model_name}")
            model_name = default_model_name

        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"❌ Failed to load embedding model '{model_name}': {e}")
            print(f"⚠️  Falling back to default model: {default_model_name}")
            self.model = SentenceTransformer(default_model_name)

        # Load FAISS index
        faiss_path = self.index_dir / "faiss_index.bin"
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

        self.index = faiss.read_index(str(faiss_path))

        # Load chunks
        chunks_path = self.index_dir / "chunks_indexed.pkl"
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        # Load chunk mapping
        mapping_path = self.index_dir / "chunk_mapping.pkl"
        with open(mapping_path, 'rb') as f:
            self.chunk_mapping = pickle.load(f)

        # Build lookup dictionaries
        self._build_lookups()

        print(f"✅ RAG system loaded successfully!")
        print(f"   - {len(self.chunks)} chunks indexed")
        print(f"   - Available chunk types: {', '.join(self.metadata.get('chunk_types', []))}")

    def _build_lookups(self):
        """Build lookup dictionaries for efficient filtering."""
        for i, chunk in enumerate(self.chunks):
            chunk_type = chunk['chunk_type']
            play_title = chunk['play_title']
            
            self.chunks_by_type[chunk_type].append((i, chunk))
            self.chunks_by_play[play_title].append((i, chunk))
    
    def classify_query(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        """Enhanced query classification with stricter rules for act, scene, and quote detection."""
        query_lower = query.lower().strip()
        
        # Define patterns with weights - ENHANCED WITH STRICTER RULES
        patterns = {
            'line_speaker': {
                'patterns': [
                    r'\bwho said\s+["\']([^"\']+)["\']',  # "who said 'quote'"
                    r'\bwho says\s+["\']([^"\']+)["\']',  # "who says 'quote'"
                    r'\bwho spoke\s+["\']([^"\']+)["\']', # "who spoke 'quote'"
                    r'\bwho uttered\s+["\']([^"\']+)["\']', # "who uttered 'quote'"
                    r'["\']([^"\']+)["\']\s*who said',    # "'quote' who said"
                    r'["\']([^"\']+)["\']\s*who spoke',   # "'quote' who spoke"
                    r'\bspeaker.*["\']([^"\']+)["\']',    # "speaker of 'quote'"
                    r'\bsaid\s+["\']([^"\']+)["\'].*who', # "said 'quote' who"
                ],
                'keywords': ['who said', 'who says', 'speaker', 'who spoke', 'who uttered'],
                'priority': 10,
                'strict_quotes': True  # Must have quotes
            },
            'scene_summary': {
                'patterns': [
                    r'\bscene\s+(\d+)[,\s]+act\s+(\d+)\b',  # "scene 1, act 1"
                    r'\bact\s+(\d+)[,\s]+scene\s+(\d+)\b',  # "act 1, scene 1"
                    r'\bscene\s+(\d+)\s+of\s+act\s+(\d+)\b', # "scene 1 of act 1"
                    r'\bact\s+(\d+)\s+scene\s+(\d+)\b',     # "act 1 scene 1"
                    r'\bwhat happens.*scene\s+(\d+)', # "what happens in scene 1"
                    r'\bsummarize.*scene\s+(\d+)',    # "summarize scene 1"
                    r'\bscene\s+([ivx]+)',            # "scene i, ii, iii"
                ],
                'keywords': ['scene', 'act scene', 'scene summary'],
                'priority': 12,
                'strict_scene': True  # Must have scene number
            },
            'act_summary': {
                'patterns': [
                    r'\bact\s+(\d+)(?!\s*scene)\b',  # "act 1" but NOT "act 1 scene"
                    r'\bsummarize\s+act\s+(\d+)(?!\s*scene)\b', # "summarize act 1" but NOT "act 1 scene"
                    r'\bwhat happens.*act\s+(\d+)(?!\s*scene)\b', # "what happens in act 1" but NOT "act 1 scene"
                    r'\bact\s+([ivx]+)(?!\s*scene)\b', # "act i, ii, iii" but NOT with scene
                    r'\bsummary.*act\s+(\d+)(?!\s*scene)\b', # "summary of act 1" but NOT with scene
                ],
                'keywords': ['act summary', 'act', 'summarize act'],
                'priority': 10,
                'strict_act': True,  # Must have act number
                'exclude_scene': True  # Must NOT have scene
            },
            'quote_to_play': {
                'patterns': [
                    r'\bwhich play.*["\']([^"\']+)["\']',    # "which play has 'quote'"
                    r'\bwhat play.*["\']([^"\']+)["\']',     # "what play contains 'quote'"
                    r'\bwho said.*["\']([^"\']+)["\']',    # "who said 'quote'"
                    r'\bwho spoke.*["\']([^"\']+)["\']',     # "who spoke 'quote'"
                    r'\bfrom which play.*["\']([^"\']+)["\']', # "from which play is 'quote'"
                    r'["\']([^"\']+)["\']\s*from\s+which\s+play', # "'quote' from which play"
                    r'["\']([^"\']+)["\']\s*which\s+play',  # "'quote' which play"
                    r'\bplay.*contains.*["\']([^"\']+)["\']', # "play contains 'quote'"
                    r'\bsource.*["\']([^"\']+)["\']',        # "source of 'quote'"
                ],
                'keywords': ['who said', 'who spoke', 'which play', 'what play', 'from play', 'source of'],
                'priority': 14,
                'strict_quotes': True  # Must have quotes
            },
            'quote_meaning': {
                'patterns': [
                    r'\bwhat does\s+["\']([^"\']+)["\']\s+mean\b',   # "what does 'quote' mean"
                    r'\bmeaning.*["\']([^"\']+)["\']',               # "meaning of 'quote'"
                    r'\bexplain.*["\']([^"\']+)["\']',               # "explain 'quote'"
                    r'\binterpret.*["\']([^"\']+)["\']',             # "interpret 'quote'"
                    r'["\']([^"\']+)["\']\s*mean',                   # "'quote' mean"
                    r'["\']([^"\']+)["\']\s*significance',           # "'quote' significance"
                ],
                'keywords': ['mean', 'meaning', 'explain', 'interpret', 'significance'],
                'priority': 13,
                'strict_quotes': True  # Must have quotes
            },
            'play_year': {
                'patterns': [
                    r'\bwhen was .* written\b',
                    r'\bwhen did .* write\b',
                    r'\bwhat year .*written',
                    r'\bwritten in \d+',
                    r'\bpublication date',
                    r'\bpublished.*when',
                    r'\bdate.*written'
                ],
                'keywords': ['when', 'year', 'written', 'published', 'date'],
                'priority': 11
            },
            'play_category': {
                'patterns': [
                    r'\bwhat (type|kind|category).*play\b',
                    r'\bis .* a (comedy|tragedy|history)\b',
                    r'\bgenre of .*\b',
                    r'\b(comedy|tragedy|history) play\b',
                    r'\bcategory.*\b',
                    r'\btype of play\b'
                ],
                'keywords': ['type', 'kind', 'category', 'genre', 'comedy', 'tragedy', 'history'],
                'priority': 11
            },
            'play_setting': {
                'patterns': [
                    r'\bwhere .* take place\b',
                    r'\bwhere .* set\b',
                    r'\bsetting of .*\b',
                    r'\blocation of .*\b',
                    r'\btakes? place in\b'
                ],
                'keywords': ['where', 'setting', 'location', 'place', 'takes place'],
                'priority': 11
            },
            'character_identity': {
                'patterns': [
                    r'\bwho is (\w+)(?!\s+said)\b',  # "who is hamlet" but NOT "who is said"
                    r'\bwho was (\w+)(?!\s+said)\b', # "who was hamlet" but NOT "who was said"
                    r'\btell me about (\w+)\b',
                    r'\bdescribe (\w+)\b',
                    r'\bcharacter (\w+)\b'
                ],
                'keywords': ['character', 'who', 'protagonist', 'person'],
                'priority': 7,
                'exclude_said': True  # Exclude if contains "said"
            },
            'main_characters': {
                'patterns': [
                    r'\bmain characters in .*\b',
                    r'\bwho are the characters\b',
                    r'\bcharacters in .*\b',
                    r'\bprotagonists in .*\b',
                    r'\bcast of characters\b'
                ],
                'keywords': ['main characters', 'characters', 'protagonists', 'cast'],
                'priority': 7
            },
            'quotes_list': {
                'patterns': [
                    r'\bquotes from .*\b',
                    r'\bfamous quotes .*\b',
                    r'\blist .* quotes\b',
                    r'\bquotations from\b'
                ],
                'keywords': ['quotes from', 'famous quotes', 'list quotes', 'quotations'],
                'priority': 6
            },
            'themes_list': {
                'patterns': [
                    r'\bthemes in .*\b',
                    r'\bmain themes .*\b',
                    r'\bwhat are the themes\b',
                    r'\bthematic elements\b'
                ],
                'keywords': ['themes', 'main themes', 'thematic'],
                'priority': 5
            },
            'theme_exploration': {
                'patterns': [
                    r'\bhow is .* theme\b',
                    r'\btheme of .* explored\b',
                    r'\bexplore theme\b',
                    r'\btheme.*developed\b'
                ],
                'keywords': ['theme explored', 'how theme', 'theme developed'],
                'priority': 5
            },
            'play_summary': {
                'patterns': [
                    r'\bsummarize (?!act|scene)\b',
                    r'\bsummary of (?!act|scene)\b',
                    r'\bwhat happens in (?!act|scene)\b',
                    r'\bplot of .*\b',
                    r'\bstory of .*\b'
                ],
                'keywords': ['summarize', 'summary', 'plot', 'story', 'what happens'],
                'priority': 4
            },
            'character_relationship': {
                'patterns': [
                    r'\brelationship between (\w+) and (\w+)',
                    r'\bhow are (\w+) and (\w+) related',
                    r'\bwhat is the relationship.*(\w+).*and.*(\w+)',
                    r'\bconnection between (\w+) and (\w+)',
                ],
                'keywords': ['relationship', 'related', 'connection', 'bond between'],
                'priority': 9,
            },

            'next_speaker': {
                'patterns': [
                    r'\bwhat is the next line after\s+["\']([^"\']+)["\']',
                    r'\bnext line after\s+["\']([^"\']+)["\']',
                    r'\bwho speaks after\s+["\']([^"\']+)["\']',
                    r'\bwhat happens after\s+["\']([^"\']+)["\']'
                ],
                'keywords': ['next line after', 'next speaker after', 'who speaks after', 'after this line'],
                'priority': 8,
                'strict_quotes': True  # ✅ Must have a quoted line
            }
        }
        
        type_scores = {}
        
        # Enhanced scoring with strict validation
        for chunk_type, data in patterns.items():
            score = 0.0
            priority = data.get('priority', 1)
            
            # STRICT VALIDATION CHECKS
            valid_for_type = True
            
            # Check if quotes are required but missing
            if data.get('strict_quotes', False):
                if not re.search(r'["\']([^"\']+)["\']', query_lower):
                    valid_for_type = False
            
            # Check if scene number is required
            if data.get('strict_scene', False):
                if not re.search(r'\bscene\s+(\d+|[ivx]+)', query_lower):
                    valid_for_type = False
            
            # Check if act number is required without scene
            if data.get('strict_act', False):
                if not re.search(r'\bact\s+(\d+|[ivx]+)', query_lower):
                    valid_for_type = False
                
                # Exclude if scene is also mentioned (for act_summary)
                if data.get('exclude_scene', False):
                    if re.search(r'\bscene\s+(\d+|[ivx]+)', query_lower):
                        valid_for_type = False
            
            # Exclude if contains "said" for character_identity
            if data.get('exclude_said', False):
                if 'said' in query_lower or 'says' in query_lower:
                    valid_for_type = False
            
            if not valid_for_type:
                type_scores[chunk_type] = 0.0
                continue
            
            # Check regex patterns (highest weight with priority)
            pattern_matched = False
            for pattern in data['patterns']:
                if re.search(pattern, query_lower):
                    score += priority * 3.0  # Higher base score for pattern match
                    pattern_matched = True
                    break
            
            # Only add keyword score if no pattern matched
            if not pattern_matched:
                for keyword in data['keywords']:
                    if keyword in query_lower:
                        score += priority * 0.5
                        break
            
            type_scores[chunk_type] = score
        
        # Get types with scores > 0, sorted by score
        relevant_types = [
            chunk_type for chunk_type, score in sorted(type_scores.items(), 
                                                      key=lambda x: x[1], 
                                                      reverse=True)
            if score > 0
        ]
        
        # If no matches, return most general types
        if not relevant_types:
            relevant_types = ['play_summary', 'character_identity', 'main_characters']
        
        return relevant_types[:1], type_scores  # Return top 3 types
    
    def extract_play_names(self, query: str) -> List[str]:
        """Enhanced play name extraction with fuzzy matching."""
        # Get all unique play titles from chunks
        all_play_titles = set(chunk['play_title'] for chunk in self.chunks)
        
        query_lower = query.lower()
        found_plays = []
        
        # Direct matching first
        for play in all_play_titles:
            play_lower = play.lower()
            if play_lower in query_lower:
                found_plays.append(play)
        
        # If no direct match, try fuzzy matching for common variations
        if not found_plays:
            play_variations = {
                'hamlet': ['hamlet'],
                'othello': ['othello'],
                'macbeth': ['macbeth'],
                'romeo': ['romeo and juliet'],
                'juliet': ['romeo and juliet'],
                'lear': ['king lear'],
                'tempest': ['the tempest'],
                'midsummer': ["a midsummer night's dream"],
                'julius': ['julius caesar'],
                'caesar': ['julius caesar'],
                'merchant': ['the merchant of venice'],
                'venice': ['the merchant of venice'],
                'twelfth': ['twelfth night'],
                'much ado': ['much ado about nothing'],
                'henry iv': ['henry iv, part 1', 'henry iv, part 2'],
                'henry v': ['henry v'],
                'henry vi': ['henry vi, part 1', 'henry vi, part 2', 'henry vi, part 3'],
                'richard ii': ['richard ii'],
                'richard iii': ['richard iii'],
                'antony': ['antony and cleopatra'],
                'cleopatra': ['antony and cleopatra'],
                'coriolanus': ['coriolanus'],
                'cymbeline': ['cymbeline'],
                'titus': ['titus andronicus'],
                'troilus': ['troilus and cressida'],
                'cressida': ['troilus and cressida'],
                'measure': ['measure for measure'],
                'shrew': ['the taming of the shrew'],
                'alls well': ["all's well that ends well"],
                'helena': ["all's well that ends well"],
                'winter': ["the winter's tale"],
                'comedy of errors': ['the comedy of errors'],
                'errors': ['the comedy of errors'],
                'as you like': ['as you like it'],
                'merry wives': ['the merry wives of windsor'],
                'windsor': ['the merry wives of windsor'],
                'two gentlemen': ['the two gentlemen of verona'],
                'verona': ['the two gentlemen of verona'],
                'pericles': ['pericles, prince of tyre'],
                'timon': ['timon of athens'],
                'john': ['king john'],
                'love labour': ["love's labour's lost"],
                'dream': ["a midsummer night's dream"],
                'taming': ['the taming of the shrew']
            }
            
            for key, variations in play_variations.items():
                if key in query_lower:
                    for variation in variations:
                        # Find exact case match from actual play titles
                        for actual_play in all_play_titles:
                            if actual_play.lower() == variation.lower():
                                found_plays.append(actual_play)
                                break
                    if found_plays:  # Break after first match
                        break
        
        return found_plays
    
    def extract_quotes(self, query: str) -> List[str]:
        """Extract quoted text from query with better patterns."""
        # Enhanced quote patterns
        quote_patterns = [
            r'["\']([^"\']+)["\']',  # Basic quotes
            r'"([^"]+)"',            # Double quotes only
            r"'([^']+)'",            # Single quotes only
            r'["""]([^"""]+)["""]',  # Curly quotes
        ]
        
        quotes = []
        for pattern in quote_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                # Clean up the quote
                cleaned_quote = match.strip()
                if len(cleaned_quote) > 3:  # Ignore very short quotes
                    quotes.append(cleaned_quote)
        
        return quotes
    
    def extract_act_scene_numbers(self, query: str) -> Tuple[str, str]:
        """Extract act and scene numbers from query."""
        query_lower = query.lower()
        
        act_num = None
        scene_num = None
        
        # Extract act number
        act_patterns = [
            r'\bact\s+(\d+)',
            r'\bact\s+([ivx]+)',
        ]
        
        for pattern in act_patterns:
            match = re.search(pattern, query_lower)
            if match:
                act_num = match.group(1)
                break
        
        # Extract scene number
        scene_patterns = [
            r'\bscene\s+(\d+)',
            r'\bscene\s+([ivx]+)',
        ]
        
        for pattern in scene_patterns:
            match = re.search(pattern, query_lower)
            if match:
                scene_num = match.group(1)
                break
        
        return act_num, scene_num
    
    def keyword_filter_chunks(self, chunks: List[Tuple[int, Dict]], query: str, quotes: List[str]) -> List[Tuple[int, Dict, List[str]]]:
        """Filter chunks using keyword matching."""
        query_words = set(query.lower().split())
        filtered_chunks = []
        
        for idx, chunk in chunks:
            content_lower = chunk['content'].lower()
            match_reasons = []
            score = 0
            
            # Check for quote matches (highest priority)
            for quote in quotes:
                if quote.lower() in content_lower:
                    score += 10
                    match_reasons.append(f"Contains quote: '{quote}'")
            
            # Check for word overlaps
            content_words = set(content_lower.split())
            overlap = query_words.intersection(content_words)
            
            if overlap:
                score += len(overlap)
                if len(overlap) > 2:
                    match_reasons.append(f"High word overlap: {len(overlap)} words")
            
            # Check play title match
            if chunk['play_title'].lower() in query.lower():
                score += 5
                match_reasons.append("Play title match")
            
            if score > 0:
                filtered_chunks.append((idx, chunk, match_reasons, score))
        
        # Sort by score
        filtered_chunks.sort(key=lambda x: x[3], reverse=True)
        return [(idx, chunk, reasons) for idx, chunk, reasons, score in filtered_chunks]
    
    def search(
        self, 
        query: str, 
        k: int = 10, 
        chunk_types: List[str] = None,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Enhanced search with play-aware filtering."""
        
        # Extract play names, quotes, and act/scene numbers
        play_names = self.extract_play_names(query)
        quotes = self.extract_quotes(query)
        act_num, scene_num = self.extract_act_scene_numbers(query)
        
        print(f"Debug - Detected plays: {play_names}")
        print(f"Debug - Detected quotes: {quotes}")
        print(f"Debug - Act/Scene: {act_num}/{scene_num}")
        print(f"Debug - Target chunk types: {chunk_types}")
        
        # Stage 1: Get candidates from FAISS
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k * 10)  # Get more for filtering
        
        all_candidates = []
        
        # Stage 2: If specific plays are mentioned, prioritize those chunks
        if play_names:
            for play_name in play_names:
                if play_name in self.chunks_by_play:
                    play_chunks = self.chunks_by_play[play_name]
                    
                    # Filter by chunk type if specified
                    if chunk_types:
                        play_chunks = [(idx, chunk) for idx, chunk in play_chunks 
                                     if chunk['chunk_type'] in chunk_types]
                    
                    # Further filter by act/scene if specified
                    if act_num or scene_num:
                        play_chunks = self.filter_by_act_scene(play_chunks, act_num, scene_num)
                    
                    # Apply keyword filtering and scoring
                    filtered_play_chunks = self.keyword_filter_chunks(play_chunks, query, quotes)
                    
                    # Add play-specific candidates with high priority
                    for idx, chunk, reasons in filtered_play_chunks[:k//len(play_names) + 2]:
                        relevance_score = 3.0  # High base score for play match
                        
                        # Boost for quotes
                        for quote in quotes:
                            if quote.lower() in chunk['content'].lower():
                                relevance_score += 2.0
                        
                        # Boost for chunk type match
                        if chunk_types and chunk['chunk_type'] in chunk_types:
                            relevance_score += 1.0
                        
                        # Boost for act/scene match
                        if act_num and self.matches_act_scene(chunk, act_num, scene_num):
                            relevance_score += 1.5
                        
                        result = SearchResult(
                            chunk_id=chunk['chunk_id'],
                            chunk_type=chunk['chunk_type'],
                            play_title=chunk['play_title'],
                            content=chunk['content'],
                            metadata=chunk['metadata'],
                            distance=0.1,  # Low distance for high relevance
                            relevance_score=relevance_score,
                            match_reasons=reasons + [f"Play match: {play_name}"]
                        )
                        all_candidates.append(result)
        
        # Stage 3: If no play-specific results and we have act/scene, search all chunks
        if not all_candidates and (act_num or scene_num) and chunk_types:
            print("Debug - No play-specific results, searching all chunks for act/scene")
            all_chunks_list = [(i, chunk) for i, chunk in enumerate(self.chunks)]
            
            # Filter by chunk type
            type_filtered = [(idx, chunk) for idx, chunk in all_chunks_list 
                           if chunk['chunk_type'] in chunk_types]
            
            # Filter by act/scene
            act_scene_filtered = self.filter_by_act_scene(type_filtered, act_num, scene_num)
            
            # Apply keyword filtering
            keyword_filtered = self.keyword_filter_chunks(act_scene_filtered, query, quotes)
            
            # Add to candidates
            for idx, chunk, reasons in keyword_filtered[:k]:
                relevance_score = 2.0  # Good score for act/scene match
                
                # Boost for quotes
                for quote in quotes:
                    if quote.lower() in chunk['content'].lower():
                        relevance_score += 1.5
                
                result = SearchResult(
                    chunk_id=chunk['chunk_id'],
                    chunk_type=chunk['chunk_type'],
                    play_title=chunk['play_title'],
                    content=chunk['content'],
                    metadata=chunk['metadata'],
                    distance=0.2,
                    relevance_score=relevance_score,
                    match_reasons=reasons + [f"Act/Scene match: {act_num}/{scene_num}"]
                )
                all_candidates.append(result)
        
        # Stage 4: Add FAISS results (with lower priority if other matches found)
        base_relevance = 0.5 if (play_names or all_candidates) else 1.0
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            chunk = self.chunks[idx]
            
            # Skip if chunk type filtering is active and this doesn't match
            if chunk_types and chunk['chunk_type'] not in chunk_types:
                continue
            
            # Skip if we already have this chunk
            if any(r.chunk_id == chunk['chunk_id'] for r in all_candidates):
                continue
            
            # Skip if act/scene filtering is active and this doesn't match
            if (act_num or scene_num) and not self.matches_act_scene(chunk, act_num, scene_num):
                continue
            
            # Calculate relevance score
            relevance_score = base_relevance / (1.0 + distance)
            
            # Check for keyword matches
            match_reasons = []
            for quote in quotes:
                if quote.lower() in chunk['content'].lower():
                    relevance_score += 1.0
                    match_reasons.append(f"Contains quote: '{quote}'")
            
            # Play title bonus
            if play_names:
                for play_name in play_names:
                    if chunk['play_title'].lower() == play_name.lower():
                        relevance_score += 2.0
                        match_reasons.append(f"Play match: {play_name}")
            
            # Act/scene bonus
            if act_num and self.matches_act_scene(chunk, act_num, scene_num):
                relevance_score += 1.5
                match_reasons.append(f"Act/Scene match: {act_num}/{scene_num}")
            
            # Apply special metadata match for relationship
            if 'character_relationship' in chunk_types:
                char_pairs = re.findall(r'\b(\w+)\s+(?:and|&)\s+(\w+)', query.lower())
                for char1, char2 in char_pairs:
                    all_candidates = [
                        r for r in all_candidates if (
                            r.chunk_type != 'character_relationship' or
                            (char1 in r.metadata.get('character_1', '').lower() and char2 in r.metadata.get('character_2', '').lower()) or
                            (char2 in r.metadata.get('character_1', '').lower() and char1 in r.metadata.get('character_2', '').lower())
                        )
                    ]

            # Match next_speaker if line or name detected
            if 'next_speaker' in chunk_types:
                quotes = self.extract_quotes(query)
                if quotes:
                    quote = quotes[0].lower()
                    all_candidates = [
                        r for r in all_candidates
                        if r.chunk_type != 'next_speaker'
                        or quote in r.metadata.get('current_line', '').lower()
                    ]
                        
            result = SearchResult(
                chunk_id=chunk['chunk_id'],
                chunk_type=chunk['chunk_type'],
                play_title=chunk['play_title'],
                content=chunk['content'],
                metadata=chunk['metadata'],
                distance=distance,
                relevance_score=relevance_score,
                match_reasons=match_reasons
            )
            
            all_candidates.append(result)
        
        # Stage 5: Rerank and return top k
        if rerank:
            all_candidates = self.rerank_results(query, all_candidates, quotes, play_names)
        
        return all_candidates[:k]
    
    def filter_by_act_scene(self, chunks: List[Tuple[int, Dict]], act_num: str, scene_num: str) -> List[Tuple[int, Dict]]:
        """Filter chunks by act and scene numbers."""
        if not act_num and not scene_num:
            return chunks
        
        filtered = []
        for idx, chunk in chunks:
            content = chunk['content'].lower()
            
            # Check if this chunk matches the requested act/scene
            if self.matches_act_scene(chunk, act_num, scene_num):
                filtered.append((idx, chunk))
        
        return filtered
    
    def matches_act_scene(self, chunk: Dict, act_num: str, scene_num: str) -> bool:
        """Check if a chunk matches the specified act and scene numbers."""
        content = chunk['content'].lower()
        
        # Convert roman numerals to arabic if needed
        def normalize_number(num_str):
            if not num_str:
                return None
            
            roman_to_arabic = {
                'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
                'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10'
            }
            
            return roman_to_arabic.get(num_str.lower(), num_str)
        
        act_num_norm = normalize_number(act_num)
        scene_num_norm = normalize_number(scene_num)
        
        # Check for act match
        act_match = False
        if act_num_norm:
            act_patterns = [
                rf'\bact\s+{act_num_norm}\b',
                rf'\bact\s+{act_num}\b' if act_num != act_num_norm else None
            ]
            act_patterns = [p for p in act_patterns if p]
            
            for pattern in act_patterns:
                if re.search(pattern, content):
                    act_match = True
                    break
        
        # Check for scene match
        scene_match = False
        if scene_num_norm:
            scene_patterns = [
                rf'\bscene\s+{scene_num_norm}\b',
                rf'\bscene\s+{scene_num}\b' if scene_num != scene_num_norm else None
            ]
            scene_patterns = [p for p in scene_patterns if p]
            
            for pattern in scene_patterns:
                if re.search(pattern, content):
                    scene_match = True
                    break
        
        # If both act and scene specified, both must match
        if act_num and scene_num:
            return act_match and scene_match
        # If only act specified
        elif act_num:
            return act_match
        # If only scene specified
        elif scene_num:
            return scene_match
        
        return True  # No filtering if neither specified
    
    def rerank_results(self, query: str, candidates: List[SearchResult], quotes: List[str], play_names: List[str]) -> List[SearchResult]:
        """Advanced reranking of search results based on multiple factors."""
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for result in candidates:
            content_lower = result.content.lower()
            
            # Base relevance score adjustment
            adjusted_score = result.relevance_score
            
            # 1. Exact quote match bonus (highest priority)
            quote_bonus = 0
            for quote in quotes:
                if quote.lower() in content_lower:
                    # Exact match
                    if quote.lower() == content_lower.strip():
                        quote_bonus += 5.0
                    # Substring match
                    else:
                        quote_bonus += 3.0
                        # Bonus for quote at beginning or end
                        if content_lower.startswith(quote.lower()) or content_lower.endswith(quote.lower()):
                            quote_bonus += 1.0
            
            # 2. Semantic similarity bonus
            content_words = set(content_lower.split())
            word_overlap = len(query_words.intersection(content_words))
            overlap_ratio = word_overlap / max(len(query_words), 1)
            semantic_bonus = overlap_ratio * 2.0
            
            # 3. Content length penalty (prefer focused content)
            length_penalty = 0
            if len(result.content) > 1000:
                length_penalty = -0.5
            elif len(result.content) < 50:
                length_penalty = -0.3
            
            # 4. Chunk type relevance (based on query classification)
            type_bonus = 0
            if result.chunk_type in ['line_speaker', 'quote_to_play', 'quote_meaning']:
                if quotes:  # Query has quotes
                    type_bonus += 1.5
            elif result.chunk_type in ['scene_summary', 'act_summary']:
                if 'scene' in query_lower or 'act' in query_lower:
                    type_bonus += 1.0
            elif result.chunk_type == 'character_identity':
                if 'who is' in query_lower or 'character' in query_lower:
                    type_bonus += 1.0
            
            # 5. Play title exact match bonus
            play_bonus = 0
            for play_name in play_names:
                if result.play_title.lower() == play_name.lower():
                    play_bonus += 2.0
                    break
            
            # 6. Recency bias for summaries (prefer earlier acts/scenes for context)
            recency_bonus = 0
            if result.chunk_type in ['act_summary', 'scene_summary']:
                # Extract act/scene numbers from content
                act_match = re.search(r'act\s+(\d+)', content_lower)
                if act_match:
                    act_num = int(act_match.group(1))
                    # Slight preference for earlier acts (more foundational)
                    recency_bonus = max(0, (6 - act_num)) * 0.1
            
            # Apply all adjustments
            adjusted_score += quote_bonus + semantic_bonus + type_bonus + play_bonus + recency_bonus + length_penalty
            
            # Update the result
            result.relevance_score = adjusted_score
            
            # Add reranking reasons
            if quote_bonus > 0:
                result.match_reasons.append(f"Quote match bonus: +{quote_bonus:.1f}")
            if semantic_bonus > 0.5:
                result.match_reasons.append(f"High semantic similarity: +{semantic_bonus:.1f}")
            if type_bonus > 0:
                result.match_reasons.append(f"Relevant chunk type: +{type_bonus:.1f}")
            if play_bonus > 0:
                result.match_reasons.append(f"Exact play match: +{play_bonus:.1f}")
        
        # Sort by adjusted relevance score
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        return candidates
    
    def format_results(self, results: List[SearchResult], show_debug: bool = False) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            header = f"[{i}] {result.play_title} ({result.chunk_type})"
            content = result.content[:500] + "..." if len(result.content) > 500 else result.content
            
            section = f"{header}\n{'-' * len(header)}\n{content}"
            
            if show_debug:
                debug_info = f"\nDEBUG: Score={result.relevance_score:.2f}, Distance={result.distance:.3f}"
                if result.match_reasons:
                    debug_info += f"\nReasons: {'; '.join(result.match_reasons)}"
                section += debug_info
            
            formatted.append(section)
        
        return "\n\n".join(formatted)
    
    def intelligent_search(self, query: str, k: int = 5, show_debug: bool = False) -> str:
        """Main search interface with intelligent query processing."""
        
        print(f"🔍 Processing query: '{query}'")
        
        # Step 1: Classify the query to determine optimal chunk types
        relevant_types, type_scores = self.classify_query(query)
        
        if show_debug:
            print(f"Debug - Query classification scores: {type_scores}")
            print(f"Debug - Selected types: {relevant_types}")
        
        # Step 2: Perform the search
        results = self.search(
            query=query,
            k=k * 2,  # Get more results for better filtering
            chunk_types=relevant_types,
            rerank=True
        )
        
        # Step 3: Final filtering and formatting
        if len(results) > k:
            results = results[:k]
        
        return self.format_results(results, show_debug=show_debug)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded corpus."""
        if not self.chunks:
            return {}
        
        stats = {
            'total_chunks': len(self.chunks),
            'chunk_types': {},
            'plays': {},
            'total_plays': len(set(chunk['play_title'] for chunk in self.chunks))
        }
        
        # Count by chunk type
        for chunk in self.chunks:
            chunk_type = chunk['chunk_type']
            play_title = chunk['play_title']
            
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
            stats['plays'][play_title] = stats['plays'].get(play_title, 0) + 1
        
        return stats

def main():
    """Example usage of the Shakespeare RAG system."""
    try:
        # Initialize the system
        rag = ShakespeareRAGSystem()
        
        # Show system statistics
        stats = rag.get_statistics()
        print(f"\n📊 Corpus Statistics:")
        print(f"   - Total chunks: {stats['total_chunks']}")
        print(f"   - Total plays: {stats['total_plays']}")
        print(f"   - Chunk types: {list(stats['chunk_types'].keys())}")
        
        # Example queries
        example_queries = [
            "Who said 'To be or not to be'?",
            "What happens in Act 1 Scene 1 of Hamlet?",
            "Which play contains the quote 'All the world's a stage'?",
            "Who is Lady Macbeth?",
            "Summarize Romeo and Juliet",
            "What are the main themes in Othello?",
            "When was Hamlet written?",
            "What does 'Fair is foul and foul is fair' mean?"
        ]
        
        print(f"\n🎭 Shakespeare RAG System Ready!")
        print(f"Try some example queries or enter your own.")
        print(f"Type 'quit' to exit, 'examples' to see example queries.\n")
        
        while True:
            try:
                user_query = input("Ask about Shakespeare: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if user_query.lower() == 'examples':
                    print("\n📝 Example queries:")
                    for i, query in enumerate(example_queries, 1):
                        print(f"  {i}. {query}")
                    print()
                    continue
                
                if not user_query:
                    continue
                
                # Process the query
                print()
                results = rag.intelligent_search(
                    query=user_query, 
                    k=3, 
                    show_debug=False
                )
                
                print(f"\n📖 Results:\n{results}\n")
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("Make sure the index files exist in the correct directory.")

if __name__ == "__main__":
    main()
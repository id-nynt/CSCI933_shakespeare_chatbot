import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

class FAISSIndexBuilder:
    def __init__(
        self, 
        chunks_dir: str = "retrieval/chunks",
        index_dir: str = "retrieval/index",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.chunks_dir = Path(chunks_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Storage for chunks and embeddings
        self.chunks = []
        self.embeddings = None
        self.chunk_mapping = {}
        
    def load_chunks(self):
        """Load all chunks from the chunks directory."""
        chunks_file = self.chunks_dir / "all_chunks.json"
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        print("Loading chunks...")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = chunks_data
        print(f"Loaded {len(self.chunks)} chunks")
        
        # Create mapping from index to chunk metadata
        for i, chunk in enumerate(self.chunks):
            self.chunk_mapping[i] = {
                'chunk_id': chunk['chunk_id'],
                'chunk_type': chunk['chunk_type'],
                'play_title': chunk['play_title'],
                'metadata': chunk['metadata']
            }
    
    def generate_embeddings(self, batch_size: int = 32):
        """Generate embeddings for all chunks."""
        print("Generating embeddings...")
        
        # Extract content from chunks
        texts = [chunk['content'] for chunk in self.chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            all_embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        print(f"Generated embeddings shape: {self.embeddings.shape}")
    
    def build_faiss_index(self, index_type: str = "flat"):
        """Build FAISS index from embeddings."""
        print(f"Building FAISS index (type: {index_type})...")
        
        if self.embeddings is None:
            raise ValueError("Embeddings not generated yet. Call generate_embeddings() first.")
        
        # Choose index type
        if index_type == "flat":
            # Exact search (L2 distance)
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == "ivf":
            # Inverted file index for faster approximate search
            nlist = min(100, len(self.chunks) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train the index
            print("Training IVF index...")
            index.train(self.embeddings)
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World index
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        index.add(self.embeddings)
        
        self.index = index
        print(f"FAISS index built with {index.ntotal} vectors")
    
    def save_index(self):
        """Save the FAISS index and related data."""
        print("Saving index files...")
        
        # Save FAISS index
        faiss_path = self.index_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(faiss_path))
        
        # Save embeddings
        embeddings_path = self.index_dir / "embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        
        # Save chunks (indexed)
        chunks_path = self.index_dir / "chunks_indexed.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save chunk mapping
        mapping_path = self.index_dir / "chunk_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.chunk_mapping, f)
        
        # Save metadata
        metadata = {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "unknown",
            "index_type": type(self.index).__name__,
            "chunk_types": list(set(chunk['chunk_type'] for chunk in self.chunks))
        }
        
        metadata_path = self.index_dir / "index_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Index saved to: {self.index_dir}")
        print(f"Files created:")
        print(f"  - faiss_index.bin")
        print(f"  - embeddings.npy")
        print(f"  - chunks_indexed.pkl")
        print(f"  - chunk_mapping.pkl")
        print(f"  - index_metadata.json")
    
    def build_complete_index(self, index_type: str = "flat", batch_size: int = 32):
        """Complete pipeline to build and save the index."""
        try:
            self.load_chunks()
            self.generate_embeddings(batch_size=batch_size)
            self.build_faiss_index(index_type=index_type)
            self.save_index()
            print("\n✅ Index building completed successfully!")
            
        except Exception as e:
            print(f"❌ Error building index: {e}")
            raise
    
    def test_index(self, query: str = "Who is Hamlet?", k: int = 5):
        """Test the built index with a sample query."""
        print(f"\n🔍 Testing index with query: '{query}'")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        print(f"Top {k} results:")
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            chunk = self.chunks[idx]
            print(f"\n{i+1}. Distance: {distance:.4f}")
            print(f"   Type: {chunk['chunk_type']}")
            print(f"   Play: {chunk['play_title']}")
            print(f"   Content: {chunk['content'][:150]}...")

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for Shakespeare chunks")
    parser.add_argument("--chunks-dir", default="retrieval/chunks", help="Directory containing chunks")
    parser.add_argument("--index-dir", default="retrieval/index", help="Directory to save index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw"], default="flat", help="FAISS index type")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--test", action="store_true", help="Test the index after building")
    
    args = parser.parse_args()
    
    # Build index
    builder = FAISSIndexBuilder(
        chunks_dir=args.chunks_dir,
        index_dir=args.index_dir,
        model_name=args.model
    )
    
    builder.build_complete_index(
        index_type=args.index_type,
        batch_size=args.batch_size
    )
    
    # Test if requested
    if args.test:
        builder.test_index()

if __name__ == "__main__":
    main()
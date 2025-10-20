import os
import json
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel


class SimpleRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir="cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedder = SentenceTransformer(model_name)
        self.documents = []
        self.doc_embeddings = None
        self.index = None
        self.doc_hashes = []

        self._load_cache()

    def _hash_doc(self, doc):
        return hashlib.sha256(doc.encode("utf-8")).hexdigest()

    def _load_cache(self):
        """Load cache files - no locking needed for reads"""
        try:
            embeddings_path = os.path.join(self.cache_dir, "embeddings.npy")
            documents_path = os.path.join(self.cache_dir, "documents.json")
            hashes_path = os.path.join(self.cache_dir, "hashes.json")
            index_path = os.path.join(self.cache_dir, "index.faiss")

            # Only load if all required files exist (consistency check)
            required_files = [documents_path, hashes_path]
            if not all(os.path.exists(f) for f in required_files):
                self.documents = []
                self.doc_embeddings = None
                self.index = None
                self.doc_hashes = []
                return

            # Load cache files without locking (reads don't need to be atomic)
            if os.path.exists(embeddings_path):
                self.doc_embeddings = np.load(embeddings_path)

            with open(documents_path, "r") as f:
                self.documents = json.load(f)

            with open(hashes_path, "r") as f:
                self.doc_hashes = json.load(f)

            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)

            print("✅ Loaded cached documents and embeddings.")
                
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
            # If any file is corrupted/partial, reset to empty state
            self.documents = []
            self.doc_embeddings = None
            self.index = None
            self.doc_hashes = []

    def _save_cache(self):
        """Simple cache saving - back to original approach"""
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.doc_embeddings is not None:
            np.save(os.path.join(self.cache_dir, "embeddings.npy"), self.doc_embeddings)
        with open(os.path.join(self.cache_dir, "documents.json"), "w") as f:
            json.dump(self.documents, f)
        with open(os.path.join(self.cache_dir, "hashes.json"), "w") as f:
            json.dump(self.doc_hashes, f)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.cache_dir, "index.faiss"))
        print("💾 Cache saved.")

    def _normalize_vectors(self, vectors):
        """Normalize vectors to unit length for cosine similarity."""
        faiss.normalize_L2(vectors)
        return vectors

    def add_documents(self, new_docs):
        # Filter out documents already in the cache
        new_docs_to_add = []
        new_hashes = []

        for doc in new_docs:
            doc_hash = self._hash_doc(doc)
            if doc_hash not in self.doc_hashes:
                new_docs_to_add.append(doc)
                new_hashes.append(doc_hash)

        if not new_docs_to_add:
            print("ℹ️ No new documents to add.")
            return

        print(f"➕ Adding {len(new_docs_to_add)} new document(s)...")
        new_embeddings = self.embedder.encode(new_docs_to_add, convert_to_numpy=True)
        # Normalize the new embeddings
        new_embeddings = self._normalize_vectors(new_embeddings)

        if self.doc_embeddings is None:
            self.doc_embeddings = new_embeddings
        else:
            self.doc_embeddings = np.vstack([self.doc_embeddings, new_embeddings])

        if self.index is None:
            dimension = new_embeddings.shape[1]
            # Use IndexFlatIP for cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(new_embeddings)
        else:
            self.index.add(new_embeddings)

        self.documents.extend(new_docs_to_add)
        self.doc_hashes.extend(new_hashes)

        self._save_cache()

    def get_retrieved_docs(self, question, top_k=2, threshold=0.8):
        query_embedding = self.embedder.encode([question], convert_to_numpy=True)
        # Normalize query embedding
        query_embedding = self._normalize_vectors(query_embedding)
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Filter results by threshold
        filtered_docs = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold:
                filtered_docs.append(self.documents[idx])
        
        if not filtered_docs:
            print(f"⚠️ Warning: No documents found with similarity >= {threshold}")
            
        return filtered_docs
    
    def get_retrieved_index_and_similarity(self, question, top_k=2, threshold=0.8):
        query_embedding = self.embedder.encode([question], convert_to_numpy=True)
        # Normalize query embedding
        query_embedding = self._normalize_vectors(query_embedding)
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Filter results by threshold
        filtered_similarities = []
        filtered_indices = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold:
                filtered_similarities.append(sim)
                filtered_indices.append(idx)
                
        if not filtered_similarities:
            print(f"⚠️ Warning: No documents found with similarity >= {threshold}")
            return np.array([]), np.array([])
            
        return np.array(filtered_similarities), np.array(filtered_indices)

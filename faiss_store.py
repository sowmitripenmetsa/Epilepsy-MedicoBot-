import faiss
import numpy as np
import os
import pickle
from embeddings import EmbeddingGenerator


class FAISSVectorStore:
    def __init__(self, dim, index_path="faiss_index.index", metadata_path="metadata.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = self._load_index()
        self.metadata = self._load_metadata()

    def _load_index(self):
        if os.path.exists(self.index_path):
            print("Loading existing FAISS index...")
            return faiss.read_index(self.index_path)
        else:
            print("Creating new FAISS index...")
            return faiss.IndexFlatL2(self.dim)

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                return pickle.load(f)
        return {}

    def add_embeddings(self, embedded_chunks):
        vectors = np.array([item["embedding"] for item in embedded_chunks]).astype("float32")
        ids = [item["chunk_id"] for item in embedded_chunks]

        print(f"Adding {len(vectors)} vectors to FAISS...")

        self.index.add(vectors)

        # Map index position to metadata
        start_idx = len(self.metadata)
        for i, chunk in enumerate(embedded_chunks):
            self.metadata[start_idx + i] = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"]
            }

        self.save()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def search(self, query_embedding, top_k=5):
        query_vector = np.array(query_embedding).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i in indices[0]:
            if i in self.metadata:
                results.append(self.metadata[i])
        return results

embedder = EmbeddingGenerator()
embedded_chunks = embedder.generate_embeddings()

# Set dimension based on embedding size (384 for MiniLM)
store = FAISSVectorStore(dim=384)
store.add_embeddings(embedded_chunks)

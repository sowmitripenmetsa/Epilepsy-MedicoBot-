from sentence_transformers import SentenceTransformer
import numpy as np
from faiss_store import FAISSVectorStore

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", dim=384):
        self.model = SentenceTransformer(model_name)
        self.vector_store = FAISSVectorStore(dim=dim)

    def retrieve(self, query, top_k=5):
        print(f"Embedding query: {query}")
        query_embedding = self.model.encode([query])[0]

        # Searching FAISS index
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results

retriever = Retriever()
results = retriever.retrieve("What are the common symptoms of epilepsy?", top_k=1)

for res in results:
    print(f"- {res['text']}\n")

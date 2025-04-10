from sentence_transformers import SentenceTransformer
from database import PostgresChunkStore

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_chunks_from_db(self):
        db = PostgresChunkStore()
        with db.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, chunk FROM chunks;")
            rows = cur.fetchall()
        db.close()
        return [{"chunk_id": row[0], "chunk": row[1]} for row in rows]

    def generate_embeddings(self):
        chunks = self.get_chunks_from_db()
        texts = [item["chunk"] for item in chunks]
        ids = [item["chunk_id"] for item in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(texts, show_progress_bar=True)

        embedded_chunks = [
            {
                "chunk_id": cid,
                "embedding": vector.tolist(),
                "text": txt
            }
            for cid, vector, txt in zip(ids, embeddings, texts)
        ]

        return embedded_chunks


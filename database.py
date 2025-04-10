import psycopg2
from psycopg2.extras import execute_values
from file_reader import FileReader
from chunking import Chunker

class PostgresChunkStore:
    def __init__(self, dbname="postgres", user="postgres", password="xxxx(your pwd)", host="localhost", port=5432):
        self.conn = psycopg2.connect(
            dbname=dbname, user=user, password=password, host=host, port=port
        )
        self.create_table()

    def create_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    filename TEXT,
                    chunk_id TEXT UNIQUE,
                    chunk TEXT
                );
            """)
            self.conn.commit()

    def insert_chunks(self, chunk_data):
        with self.conn.cursor() as cur:
            records = [(item["filename"], item["chunk_id"], item["chunk"]) for item in chunk_data]
            execute_values(
                cur,
                """
                INSERT INTO chunks (filename, chunk_id, chunk)
                VALUES %s
                ON CONFLICT (chunk_id) DO NOTHING;
                """,
                records
            )
            self.conn.commit()

    def clear_chunks(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks;")
            self.conn.commit()

    def close(self):
        self.conn.close()

if __name__ == "__main__":

    reader = FileReader("file_path")
    data = reader.read_all_files()

    chunker = Chunker()
    chunks = chunker.chunk_all(data)

    store = PostgresChunkStore()
    store.insert_chunks(chunks)
    print(f"Inserted {len(chunks)} chunks into PostgreSQL.")
    store.close()

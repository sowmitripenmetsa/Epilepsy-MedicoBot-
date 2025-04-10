from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

class Chunker:
    def __init__(self, chunk_size=200, chunk_overlap=75):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                r"(?i)(?=\n[A-Z \-]{3,}\n)",  # ALL CAPS headings or section breaks
                r"(?i)(?=•)",                 # Bullet points
                r"(?=\n\d+\.)",              # Numbered list (1., 2., etc.)
                "\n\n",                      # Paragraphs
                "\n",                        # Line breaks
                ".",                         # Sentences
                ";", ",", "-", ":", "—", "–",
                " ",                         # Word-level
                ""                           # Character-level fallback
            ]
        )

    def chunk_single_document(self, filename, text):
        chunks = self.splitter.create_documents([text])
        return [
            {
                "filename": filename,
                "chunk_id": f"{filename}_{i}",
                "chunk": chunk.page_content
            }
            for i, chunk in enumerate(chunks)
        ]

    def chunk_all(self, file_data):
        all_chunks = []
        for filename, content in file_data:
            chunked = self.chunk_single_document(filename, content)
            all_chunks.extend(chunked)
        return all_chunks

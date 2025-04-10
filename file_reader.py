import os
from PyPDF2 import PdfReader

class FileReader:
    def __init__(self, directory="eeg_data"):
        self.directory = directory

    def read_txt(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT file {filepath}: {e}")
            return ""

    def read_pdf(self, filepath):
        try:
            reader = PdfReader(filepath)
            return "\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        except Exception as e:
            print(f"Error reading PDF file {filepath}: {e}")
            return ""

    def read_all_files(self):
        file_data = []
        for filename in os.listdir(self.directory):
            path = os.path.join(self.directory, filename)

            if filename.lower().endswith(".txt"):
                print(f"Loading TXT file: {filename}")
                content = self.read_txt(path)
            elif filename.lower().endswith(".pdf"):
                print(f"Loading PDF file: {filename}")
                content = self.read_pdf(path)
            else:
                print(f"Skipping unsupported file: {filename}")
                continue

            if content.strip():
                file_data.append((filename, content))
            else:
                print(f"Skipped empty file: {filename}")

        return file_data

if __name__=="__main__":
    reader = FileReader("/home/akshaya/Documents/medicobot/eeg_data")
    data = reader.read_all_files()
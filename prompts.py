import requests
from retriever import Retriever

class MedicalChatbot:
    def __init__(self, model="llama3.2", host="http://localhost:11434"):
        self.model = model
        self.host = host
        self.retriever = Retriever()

    def build_prompt(self, context_chunks, user_query):
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
        return f"""
You are a helpful medical assistant specialized in epilepsy.
 Instructions:
- Read and understand the context fully before answering.
- Provide **clear, medically accurate** answers.
- Limit your response to **no more than 200 words**.
- Be concise but complete — focus only on the most relevant points.
- If the answer is not directly found in the context, reply:  
  _"Based on the provided information, this detail is not available."_

Here is the clinical context:

{context_text}

Answer this question clearly and concisely:

{user_query}
""".strip()

    def answer(self, user_query):
        chunks = self.retriever.retrieve(user_query, top_k=5)
        prompt = self.build_prompt(chunks, user_query)

        print("Sending prompt to Ollama...")
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "num_predict": 200}
        )

        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"


bot = MedicalChatbot()
res = bot.answer("Can you summarize the EEG activity patterns for this patient?")
print(res)

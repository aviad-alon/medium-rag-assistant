import os
import json
from http.server import BaseHTTPRequestHandler
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# --- Configuration (must match query.py) ---
EMBED_MODEL = "NBUECSE-text-embedding-3-small"
CHAT_MODEL = "NBUECSE-gpt-5-mini"
TOP_K = 8
SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). "
    "You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: \"I don't know based on the provided Medium articles data.\"\n\n"
    "Follow these output format rules based on the type of question:\n"
    "1. If asked to find a specific article and return its title and author, use exactly:\n"
    "   Title: <title>\n"
    "   Author: <author>\n"
    "2. If asked to list multiple articles, use exactly this format for each:\n"
    "   1. Title: <title>\n"
    "      Author: <author>\n"
    "   (repeat for each article)\n"
    "3. If asked to summarise an article, answer in 2-3 sentences based only on the retrieved passages.\n"
    "4. If asked for a recommendation, include the article title, author, and a short explanation (without using the label 'Justification:') grounded in the retrieved text.\n"
    "For all other questions (e.g. simple fact extraction), answer directly and concisely without adding unnecessary fields.\n"
    "If the user specifies an exact output format, follow it strictly instead."
)

# --- Clients (initialized once per Lambda instance) ---
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        question = body.get("question", "")

        # Step 1: embed the question
        embed_response = client.embeddings.create(model=EMBED_MODEL, input=[question])
        vector = embed_response.data[0].embedding

        # Step 2: search Pinecone
        results = index.query(vector=vector, top_k=TOP_K, include_metadata=True)
        matches = results.matches

        # Step 3: build context with article headers
        chunks = []
        for match in matches:
            header = f"[Article: \"{match.metadata['title']}\" | Author: {match.metadata['author']}]"
            chunks.append(header + "\n" + match.metadata["text"])
        context = "\n\n".join(chunks)

        # Step 4: build augmented user prompt
        user_prompt = (
            f"Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n"
        )

        # Step 5: generate answer
        llm_response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = llm_response.choices[0].message.content

        # Step 6: build JSON response
        result = {
            "response": answer,
            "context": [
                {
                    "article_id": m.metadata["article_id"],
                    "title": m.metadata["title"],
                    "author": m.metadata.get("author", ""),
                    "chunk": m.metadata["text"],
                    "score": m.score,
                }
                for m in matches
            ],
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User": user_prompt,
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

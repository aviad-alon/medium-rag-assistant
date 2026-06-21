import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
EMBED_MODEL = "NBUECSE-text-embedding-3-small"
CHAT_MODEL = "NBUECSE-gpt-5-mini"
TOP_K = 8

# --- Clients ---
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


# Convert the user's question into a vector using the embedding model.
# Args:    question: The user's question as a string.
# Returns: A list of 1536 floats representing the question vector.
def embed_question(question: str) -> list:
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    embedding_vector = response.data[0].embedding
    return embedding_vector


# Search Pinecone for the TOP_K most similar chunks to the question vector.
# Args:    embedding_vector: The vector representation of the user's question.
# Returns: A list of TOP_K matches, each containing a score and metadata.
def search_pinecone(embedding_vector: list) -> list:
    results = index.query(vector=embedding_vector, top_k=TOP_K, include_metadata=True)
    return results.matches


# Extract the text from each match and join them into a single context string.
# Args:    matches: List of Pinecone match objects containing metadata.
# Returns: A single string with all chunk texts separated by double newlines.
def build_context(matches: list) -> str:
    chunks = []
    for match in matches:
        header = f"[Article: \"{match.metadata['title']}\" | Author: {match.metadata['author']}]"
        chunks.append(header + "\n" + match.metadata["text"])
    context = "\n\n".join(chunks)
    return context


# Combine the retrieved context with the original question to create an augmented prompt.
# Args:    question: The original user question.
#          context:  The retrieved context from Pinecone.
# Returns: A formatted prompt string ready to be sent to the LLM.
def build_augmented_prompt(question: str, context: str) -> str:
    augmented_prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{question}
"""
    return augmented_prompt


# Send the augmented prompt to the LLM and return its answer.
# Args:    augmented_prompt: The full prompt including context and question.
# Returns: The LLM's answer as a string.
def generate_answer(augmented_prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": (
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
                "4. If asked for a recommendation, include the article title, author, and a short justification grounded in the retrieved text.\n"
                "For all other questions (e.g. simple fact extraction), answer directly and concisely without adding unnecessary fields.\n"
                "If the user specifies an exact output format, follow it strictly instead."
            )},
            {"role": "user", "content": augmented_prompt},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    question = input("Enter your question: ")

    embedding_vector = embed_question(question)
    matches = search_pinecone(embedding_vector)
    context = build_context(matches)
    augmented_prompt = build_augmented_prompt(question, context)
    answer = generate_answer(augmented_prompt)

    print("\n--- ANSWER ---")
    print(answer)

    print("\n--- SOURCES ---")
    for match in matches:
        print(f"  Article: {match.metadata['title']} | Chunk: {match.metadata['chunk_index']}")

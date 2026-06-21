import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CSV_FILE = "medium-english-50mb.csv"
CHUNK_SIZE = 600
OVERLAP = 60  # 10% of CHUNK_SIZE
EMBED_MODEL = "NBUECSE-text-embedding-3-small"
BATCH_SIZE = 100

# --- OpenAI client (used for embeddings via llmod) ---
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# --- Pinecone client ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def load_articles(filepath: str) -> pd.DataFrame:
    """
    Load articles from a CSV file.
    Returns a DataFrame with columns: title, text, url, authors, timestamp, tags.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} articles with columns: {list(df.columns)}")
    return df



def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split a single article's text into overlapping chunks of words.

    Called once per article — does NOT process the entire dataset.
    Each call receives the raw text of one article and returns a list
    of smaller text pieces (chunks) that overlap slightly.

    Args:
        text:       The full text of a single article.
        chunk_size: Number of words per chunk (e.g. 600).
        overlap:    Number of words to repeat from the previous chunk (e.g. 60).

    Returns:
        A list of strings, each representing one chunk of the article.
        Example: ["word1 word2 ... word600", "word541 ... word1140", ...]
    """
    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


def prepare_chunks(df: pd.DataFrame) -> list:
    """
    Iterate over all articles and split each one into chunks.

    For every article in the DataFrame, calls chunk_text() and wraps
    each resulting chunk in a dict with metadata (article_id, title, chunk_index, text).

    Args:
        df: DataFrame containing all articles.

    Returns:
        A flat list of dicts, one dict per chunk across all articles.
        Example: [{"article_id": "0", "title": "...", "chunk_index": 0, "text": "..."}, 
                 [{"article_id": "0", "title": "...", "chunk_index": 1, "text": "..."}, 
                ...]
    """
    all_chunks = []

    for idx, row in df.iterrows():
        chunks = chunk_text(row["text"], CHUNK_SIZE, OVERLAP)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append({
                "article_id": str(idx),
                "title": row["title"],
                "author": row["authors"],
                "chunk_index": chunk_index,
                "text": chunk,
            })

    return all_chunks


def embed_chunks(chunks: list) -> list:
    """
    Embed all chunks using the ZYRANGG embedding model, in batches.

    Sends chunks in groups of BATCH_SIZE to avoid large single requests.
    Each chunk dict is updated in-place with a "vector" field containing
    the 1536-dimensional embedding returned by the model.

    Args:
        chunks: List of chunk dicts (must have a "text" field each).

    Returns:
        The same list of chunk dicts, each now containing a "vector" field.
    """
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(model=EMBED_MODEL, input=texts)

        for j, item in enumerate(response.data):
            batch[j]["vector"] = item.embedding

        print(f"Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    return chunks


def upsert_to_pinecone(chunks: list) -> None:
    """
    Upload embedded chunks to Pinecone in batches.

    Each chunk is stored as a vector with an ID and metadata so it can be
    retrieved and traced back to its source article later.

    The vector ID format is "{article_id}_{chunk_index}" (e.g. "0_0", "0_1", "1_0").
    Metadata includes: article_id, title, chunk_index, and the original text.

    Args:
        chunks: List of chunk dicts, each must have "vector", "article_id",
                "title", "chunk_index", and "text" fields.
    """
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectors = [
            {
                "id": f"{c['article_id']}_{c['chunk_index']}",
                "values": c["vector"],
                "metadata": {
                    "article_id": c["article_id"],
                    "title": c["title"],
                    "author": c["author"],
                    "chunk_index": c["chunk_index"],
                    "text": c["text"],
                },
            }
            for c in batch
        ]
        index.upsert(vectors=vectors)
        print(f"Upserted {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks to Pinecone")


if __name__ == "__main__":
    df = load_articles(CSV_FILE)

    # Step 1: split all articles into chunks
    all_chunks = prepare_chunks(df)
    print(f"Total chunks: {len(all_chunks)}")

    # Step 2: embed all chunks using the embedding model
    all_chunks = embed_chunks(all_chunks)

    # Step 3: upload vectors + metadata to Pinecone
    upsert_to_pinecone(all_chunks)
    print("Done! All chunks uploaded to Pinecone.")

# Medium RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions based on a dataset of 7,682 Medium articles. Answers are grounded strictly in the dataset - no external knowledge used.

**Live Demo:** [medium-rag-assistant-gilt.vercel.app](https://medium-rag-assistant-gilt.vercel.app)

---

## System Flow

```
                                         +-----------------+
                                         |    Vector DB    |
                                         |    (Pinecone)   |
                                         +-----------------+
                                            ^           |
                                  embed     |           |  top-8 chunks
                                  vector    |           v

  User Question  -->  [ Embedding ]  ------+    [ Augmented Prompt ]  -->  [ LLM ]  -->  Answer

                                                system prompt
                                               + context chunks
                                               + user question
```

---

## What the System Can Do

The assistant supports 4 types of queries, all answered strictly from the dataset:

### 1. Precise Fact Retrieval
Locate a single specific article based on semantic criteria and return its title and author.

**Example query:**
> "Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author."

**Expected output:**
```
Title: <article title>
Author: <author name>
```

---

### 2. Multi-result Topic Listing (up to 3 results)
Return multiple distinct article titles matching a theme or topic.

**Example query:**
> "List exactly 3 articles about education. Return only the titles."

**Expected output:**
```
1. Title: <title 1>
2. Title: <title 2>
3. Title: <title 3>
```

---

### 3. Key Idea Summary Extraction
Identify a relevant article and generate a concise 2-3 sentence summary of its main idea, based only on retrieved passages.

**Example query:**
> "Find an article that argues past pandemics can spur innovation and recovery, and summarise its central argument."

**Expected output:**
> A 2-3 sentence summary grounded in the retrieved text.

---

### 4. Recommendation with Evidence-Based Justification
Recommend one relevant article and explain why, grounded in the retrieved text.

**Example query:**
> "I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?"

**Expected output:**
```
Title: <article title>
Author: <author name>
<short explanation grounded in the article text>
```

---

### Out of Scope
If the answer cannot be found in the dataset, the system responds:
> "I don't know based on the provided Medium articles data."

---

## Architecture

### Ingestion (offline, one-time)

| Step | Description |
|------|-------------|
| Load | Read `medium-english-50mb.csv` - 7,682 articles |
| Chunk | Split each article into 600-word chunks with 10% overlap (60 words) |
| Embed | Encode each chunk using `text-embedding-3-small` in batches of 100 |
| Store | Upsert vectors + metadata (title, author, chunk text) into Pinecone |

### Query API (serverless, Vercel)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/prompt` | POST | Accepts `{ "question": "..." }`, returns answer + retrieved context |
| `/api/stats` | GET | Returns system configuration (chunk size, overlap, top-k) |

### Tech Stack

| Component | Technology |
|-----------|------------|
| Vector DB | Pinecone |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-5-mini` |
| Backend | Python (Vercel Serverless Functions) |
| Frontend | Vanilla HTML/CSS/JS |
| Hosting | Vercel |

---

## RAG Parameters

| Parameter | Value | Constraint |
|-----------|-------|------------|
| Chunk size | 600 words | max 1024 |
| Overlap | 60 words (10%) | max 30% |
| Top-K | 8 | max 30 |
| Embedding dimensions | 1536 | - |

---

## Project Structure

```
api/
  prompt.py      # POST /api/prompt - RAG query endpoint
  stats.py       # GET  /api/stats  - system config endpoint
ingest.py        # One-time ingestion script
query.py         # Local testing script
index.html       # Frontend UI
vercel.json      # Deployment configuration
requirements.txt # Python dependencies
```

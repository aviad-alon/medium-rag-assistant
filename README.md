# Medium RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions based on a dataset of 7,682 Medium articles.

**Live Demo:** [medium-rag-assistant-gilt.vercel.app](https://medium-rag-assistant-gilt.vercel.app)

---

## System Flow

```
User Question
     │
     ▼
┌─────────────────────┐
│   Embedding Model   │  text-embedding-3-small (via llmod.ai)
│  Question → Vector  │
└────────┬────────────┘
         │  1536-dim vector
         ▼
┌─────────────────────┐
│      Pinecone       │  Vector similarity search
│   Top-8 Retrieval   │  ~50,000 article chunks indexed
└────────┬────────────┘
         │  8 most relevant chunks + metadata
         ▼
┌─────────────────────┐
│  Augmented Prompt   │  System prompt + context + question
│  (Context Builder)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│     LLM (GPT)       │  gpt-5-mini (via llmod.ai)
│  Answer Generation  │  Strictly grounded in retrieved context
└────────┬────────────┘
         │
         ▼
     JSON Response
  { response, context, augmented_prompt }
```

---

## Architecture

### Ingestion (offline, one-time)

| Step | Description |
|------|-------------|
| Load | Read `medium-english-50mb.csv` — 7,682 articles |
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

- **Chunk size:** 600 words
- **Overlap:** 60 words (10%)
- **Top-K:** 8 chunks retrieved per query
- **Embedding dimensions:** 1536

---

## Project Structure

```
├── api/
│   ├── prompt.py      # POST /api/prompt — RAG query endpoint
│   └── stats.py       # GET  /api/stats  — system config endpoint
├── ingest.py          # One-time ingestion script
├── query.py           # Local testing script
├── index.html         # Frontend UI
├── vercel.json        # Deployment configuration
└── requirements.txt   # Python dependencies
```

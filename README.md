# 📚 RAG Document Assistant

> **LangChain · FAISS · Streamlit · AWS S3 · OpenAI**

A production-grade document Q&A assistant. Upload PDFs, ask questions in natural language, and get accurate answers grounded in your documents — with full conversation memory and AWS S3 persistence.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **PDF Upload** | Multi-file upload with page-by-page text extraction |
| **Semantic Search** | FAISS vector index + OpenAI `text-embedding-3-small` |
| **Conversational Memory** | Rolling-window history injected into every prompt |
| **AWS S3 Persistence** | PDFs stored in S3; FAISS index persisted on disk |
| **Streaming UI** | Streamlit dark-mode interface with source citations |
| **Docker + CI/CD** | Multi-stage Dockerfile + GitHub Actions → ECR → ECS |

---

## 🏗 Architecture

```
┌─────────────┐     upload      ┌────────────────────┐
│   Browser   │ ─────────────▶  │  Streamlit (8501)  │
│  (User UI)  │ ◀─────────────  │     app/main.py    │
└─────────────┘     answer      └────────┬───────────┘
                                         │
              ┌──────────────────────────┼───────────────────────┐
              │                          │                       │
              ▼                          ▼                       ▼
      ┌───────────────┐        ┌──────────────────┐    ┌──────────────────┐
      │  PDFProcessor │        │   EmbeddingMgr   │    │    S3Handler     │
      │  (pdfplumber) │──docs─▶│  FAISS + OpenAI  │    │   (boto3/AWS)    │
      └───────────────┘        └────────┬─────────┘    └──────────────────┘
                                        │ top-k chunks
                               ┌────────▼─────────┐
                               │  DocumentRetriever│
                               │  (dedup + format) │
                               └────────┬──────────┘
                                        │ context
                               ┌────────▼──────────┐
                               │     RAGChain      │
                               │  LangChain + GPT  │◀─── ConversationMemory
                               └───────────────────┘
```

---

## 📁 Project Structure

```
rag-document-assistant/
├── app/
│   ├── main.py                 # Streamlit UI entry point
│   ├── rag/
│   │   ├── chain.py            # LLM chain (retrieve → prompt → generate)
│   │   ├── embeddings.py       # FAISS vector store manager
│   │   ├── memory.py           # Conversation history buffer
│   │   └── retriever.py        # Query → top-k chunks + formatting
│   ├── storage/
│   │   └── s3_handler.py       # AWS S3 upload / download / list
│   └── utils/
│       ├── config.py           # pydantic-settings config singleton
│       ├── logger.py           # Structured logging setup
│       └── pdf_processor.py    # PDF parse → chunk → Document[]
├── tests/
│   ├── test_rag.py             # Unit tests: memory, retriever, processor
│   └── test_storage.py         # Unit tests: S3 (moto mock)
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI: lint → test → build → ECS deploy
├── scripts/
│   ├── setup.sh                # Local dev bootstrap
│   └── deploy_ec2.sh           # One-click EC2 deploy
├── .streamlit/
│   └── config.toml             # Streamlit dark theme config
├── .env.example                # All required environment variables
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Local Docker Compose
├── pyproject.toml              # Black / isort / pytest config
└── requirements.txt
```

---

## 🚀 Quick Start — Local Development

### 1. Clone & setup

```bash
git clone https://github.com/YOUR_USER/rag-document-assistant.git
cd rag-document-assistant

bash scripts/setup.sh
```

### 2. Configure environment

```bash
# Edit .env — the ONLY required key is OPENAI_API_KEY
nano .env
```

Minimum required `.env`:
```env
OPENAI_API_KEY=sk-...your-key...
```

AWS S3 (optional — the app works without it):
```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=my-rag-bucket
```

### 3. Run the app

```bash
source .venv/bin/activate
streamlit run app/main.py
```

Open **http://localhost:8501**

---

## 🐳 Docker

### Build and run with Docker Compose

```bash
cp .env.example .env   # fill in your keys
docker compose up --build
```

### Manual Docker commands

```bash
# Build
docker build -t rag-document-assistant:latest .

# Run
docker run -d \
  --name rag-assistant \
  -p 8501:8501 \
  --env-file .env \
  -v rag_faiss:/app/data \
  rag-document-assistant:latest
```

---

## ☁️ AWS Deployment (ECS via GitHub Actions)

### Prerequisites

1. **ECR Repository** — Create in AWS console or:
   ```bash
   aws ecr create-repository --repository-name rag-document-assistant
   ```

2. **ECS Cluster + Service** — Use Fargate with the Docker image.

3. **IAM OIDC Role** — Grant GitHub Actions permission to push to ECR and update ECS.

4. **GitHub Secrets** — Add to your repo (`Settings → Secrets`):

   | Secret | Value |
   |---|---|
   | `AWS_ROLE_ARN` | ARN of the OIDC role |
   | `ECS_CLUSTER` | Name of your ECS cluster |
   | `ECS_SERVICE` | Name of your ECS service |

### Deploy flow

```
push to main
  └─▶ CI: lint + pytest
         └─▶ docker build → push to ECR
                └─▶ aws ecs update-service (rolling deploy)
```

### EC2 quick deploy

```bash
# SSH into EC2 instance, then:
bash scripts/deploy_ec2.sh
```

---

## 🧪 Running Tests

```bash
source .venv/bin/activate

# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=term-missing
```

Tests use `moto` to mock AWS S3 — no real AWS credentials required.

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | **required** | Your OpenAI API key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_TEMPERATURE` | `0.2` | LLM temperature (0–1) |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `FAISS_INDEX_PATH` | `./data/faiss_index` | Where to persist the index |
| `MAX_HISTORY_MESSAGES` | `10` | Rolling conversation window |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max PDF file size |
| `AWS_REGION` | `us-east-1` | S3 region |
| `S3_BUCKET_NAME` | `rag-document-assistant` | S3 bucket for PDFs |

---

## 🛠 Tech Stack

- **[LangChain](https://python.langchain.com/)** — RAG orchestration, prompt management, LLM abstraction
- **[FAISS](https://github.com/facebookresearch/faiss)** — High-performance vector similarity search (CPU)
- **[OpenAI](https://platform.openai.com/)** — `text-embedding-3-small` for embeddings, GPT-4o-mini for generation
- **[Streamlit](https://streamlit.io/)** — Interactive web UI with file upload and chat
- **[pdfplumber](https://github.com/jsvine/pdfplumber)** — Accurate PDF text extraction
- **[boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)** — AWS S3 persistence
- **[pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)** — Type-safe configuration management
- **[moto](https://docs.getmoto.org/)** — AWS mock for unit tests

---

MIT © 2024

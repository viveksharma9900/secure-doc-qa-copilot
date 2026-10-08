# Secure Document Q&A Copilot (RAG + Semantic Kernel + Azure OpenAI)

Upload documents, ask questions in a React chat UI, get answers **only from your documents, with sources**.
Personal data (emails, phones, Aadhaar, PAN, cards) is **masked before it is stored or sent to the LLM**.

## Run it (3 commands, no Azure needed)
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://localhost:8000. A sample HR handbook is preloaded. Try: "How many leave days do I get?"
API docs (auto-generated): http://localhost:8000/docs

## Switch on Azure OpenAI + Semantic Kernel
1. Azure Portal: create an Azure OpenAI resource; in AI Foundry / AI Studio deploy one chat model and one embedding model.
2. `cp .env.example .env` and fill the 4 values. Restart. The badge in the UI changes to "Azure OpenAI + Semantic Kernel".

## Architecture
React UI -> FastAPI REST (`/upload`, `/chat`, `/documents`, `/health`)
Ingestion: extract text -> **mask PII** -> chunk (700 chars, 20-word overlap) -> embed -> store (tenant tagged)
Chat: mask question -> retrieve top-4 for this tenant -> Semantic Kernel prompt (grounded, cite sources) -> Azure OpenAI

| File | Purpose |
|---|---|
| `app/pii.py` | Regex PII masking (email, Indian phone, Aadhaar, PAN, card) |
| `app/chunking.py` | Paragraph-aware chunking with overlap |
| `app/store.py` | Vector store with tenant isolation; Azure embeddings or local fallback |
| `app/llm.py` | Semantic Kernel kernel + grounded prompt template; demo-mode fallback |
| `app/plugins.py` | SK plugins (`RetrievalPlugin`, `PrivacyPlugin`) using `@kernel_function` |
| `app/main.py` | FastAPI app, validation, size/type limits, session memory |
| `static/index.html` | React chat UI (React via CDN, no build step) |
| `tests/test_core.py` | pytest tests: masking, chunking, retrieval, tenant isolation |

Run tests: `pytest`   Docker: `docker build -t docqa . && docker run -p 8000:8000 --env-file .env docqa`

## Honest limitations (say these in the interview, they show maturity)
- Vector store is in-memory; production swap is Azure AI Search (same `add/search` interface).
- PII masking is regex-based; production upgrade is Azure AI Language PII detection (catches names, addresses).
- Demo mode answers by extracting sentences; real generation needs Azure OpenAI keys.
- Sessions are in memory; production would use Redis or Cosmos DB.

## Interview script (2 minutes)
"I built a secure document Q&A copilot. Documents are uploaded through a FastAPI endpoint, PII is masked before anything is stored, text is chunked with overlap and embedded, and kept in a tenant-filtered vector store. At question time I retrieve the top four chunks and a Semantic Kernel prompt makes Azure OpenAI answer only from that context and cite the source, otherwise say it doesn't know. The front end is React. I wrote tests for masking, chunking, retrieval and tenant isolation. The in-memory store is a deliberate simplification; I'd move to Azure AI Search and Azure AI Language for production."

## Likely follow-ups and your answers
- **Why RAG, not fine-tuning?** Cheaper, instantly updatable, gives citations, reduces hallucination.
- **Why chunk overlap?** Keeps sentences that straddle a boundary retrievable.
- **How do you stop hallucination?** Grounded prompt, "say I don't know", low temperature, cite sources, retrieval threshold.
- **How is privacy handled?** Mask before storage and before the LLM, tenant filter, no raw PII in logs, secrets in env/Key Vault.
- **Prompt injection?** Context is labeled as data in the prompt; no tools are triggered by document text.
- **How would you scale it?** Stateless containers, Azure AI Search, Redis sessions, queue for ingestion, rate limiting.
- **How would you evaluate it?** Build 25 Q&A pairs, measure retrieval hit-rate and answer faithfulness, compare chunk sizes.

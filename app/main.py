import io
import pathlib
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, llm
from .chunking import chunk_text
from .pii import mask_pii
from .store import VectorStore

BASE = pathlib.Path(__file__).resolve().parent.parent
app = FastAPI(title="Secure Document Q&A Copilot", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000"], allow_methods=["*"], allow_headers=["*"])

store = VectorStore()
sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    session_id: str = "default"
    question: str = Field(min_length=2, max_length=1000)


def ingest(text: str, filename: str, tenant: str) -> dict:
    masked, pii_counts = mask_pii(text)                                          # 1. mask PII first
    chunks = chunk_text(masked, config.CHUNK_CHARS, config.CHUNK_OVERLAP_WORDS)  # 2. chunk
    n = store.add(chunks, filename, tenant)                                      # 3. embed + store
    return {"file": filename, "chunks": n, "pii_masked": pii_counts}


def read_upload(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="ignore")


@app.on_event("startup")
def load_samples():
    for f in (BASE / "sample_docs").glob("*.txt"):
        ingest(f.read_text(encoding="utf-8"), f.name, "demo")


@app.get("/health")
def health():
    return {"status": "ok", "mode": "azure" if config.AZURE_CHAT_ON else "demo",
            "embeddings": "azure" if config.AZURE_EMBED_ON else "local"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), x_tenant: str = Header("demo")):
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(400, f"Only {', '.join(sorted(config.ALLOWED_EXT))} files are allowed.")
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    text = read_upload(file.filename, data)
    if not text.strip():
        raise HTTPException(422, "No readable text found in the file.")
    return ingest(text, file.filename, x_tenant)


@app.get("/documents")
def documents(x_tenant: str = Header("demo")):
    return {"documents": store.sources(x_tenant)}


@app.post("/chat")
async def chat(req: ChatRequest, x_tenant: str = Header("demo")):
    q_masked, _ = mask_pii(req.question)  # never send user PII to the model
    hits = store.search(q_masked, x_tenant, config.TOP_K)
    history = sessions.setdefault(f"{x_tenant}:{req.session_id}", [])
    reply = await llm.answer(q_masked, hits, history)
    history += [{"role": "user", "content": q_masked}, {"role": "assistant", "content": reply}]
    del history[:-12]  # cap memory
    return {"answer": reply,
            "sources": [{"source": h["source"], "score": h["score"], "snippet": h["text"][:160]} for h in hits]}


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

"""Vector store with tenant isolation.
Embeddings: Azure OpenAI if configured, else a local hashed bag-of-words vector (demo mode).
Swap this class for Azure AI Search in production (same add/search interface)."""
import math
import re
import hashlib
from . import config

STOP = set("a an the is are was were of to in on for and or with by at from as it this that be can do does what how".split())
DIM = 512


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _local_embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for t in _tokens(text):
        vec[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIM] += 1.0
    return _norm(vec)


def embed(texts: list[str]) -> list[list[float]]:
    if config.AZURE_EMBED_ON:
        from openai import AzureOpenAI
        client = AzureOpenAI(azure_endpoint=config.ENDPOINT, api_key=config.API_KEY, api_version="2024-06-01")
        res = client.embeddings.create(model=config.EMBED_DEPLOYMENT, input=texts)
        return [_norm(d.embedding) for d in res.data]
    return [_local_embed(t) for t in texts]


class VectorStore:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, chunks: list[str], source: str, tenant: str) -> int:
        for chunk, vec in zip(chunks, embed(chunks)):
            self.rows.append({"text": chunk, "vec": vec, "source": source, "tenant": tenant})
        return len(chunks)

    def search(self, query: str, tenant: str, k: int = 4) -> list[dict]:
        q = embed([query])[0]
        scored = []
        for r in self.rows:
            if r["tenant"] != tenant:  # tenant isolation: users never see others' data
                continue
            scored.append((sum(a * b for a, b in zip(q, r["vec"])), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text": r["text"], "source": r["source"], "score": round(s, 3)}
                for s, r in scored[:k] if s > 0.05]

    def sources(self, tenant: str) -> list[str]:
        return sorted({r["source"] for r in self.rows if r["tenant"] == tenant})

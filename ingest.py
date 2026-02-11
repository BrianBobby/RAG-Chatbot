# ingest.py
# Run ONCE to parse PDF -> chunk -> embed -> upsert into Pinecone.
# Requirements:
# pip install langchain-community pypdf langchain_openai python-dotenv tenacity pinecone-client

import os
import re
import hashlib
import math
from time import sleep
from typing import List
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------- CONFIG ----------
PDF_PATH = os.getenv("PDF_PATH", "Citi_Business_client Manual.pdf")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", None)    # e.g. "us-east-1"
INDEX_NAME = os.getenv("PINECONE_INDEX", "citibusiness-manual")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")         # optional
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")
BATCH_EMBED = int(os.getenv("BATCH_EMBED", 64))
BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", 64))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
# ----------------------------

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY must be set in .env")

# -------------------------
# Load and chunk the PDF
# -------------------------
def load_and_chunk(pdf_path: str) -> List[Document]:
    print(f"> Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Clean page-level artifacts
    cleaned_pages = []
    for doc in pages:
        text = doc.page_content or ""
        text = re.sub(r"CitiBusiness®\s+Client Manual.*?\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\s*\d+\s*$", "\n", text)
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        cleaned_pages.append(text)

    full_text = "\n\n".join(cleaned_pages)

    # Split into major sections (if present)
    section_pattern = re.compile(r"(?m)^(\d+\.\s+[A-Z][^\n]+)")
    matches = list(section_pattern.finditer(full_text))
    sections = []
    if matches:
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
            title = match.group(1).strip()
            body = full_text[start:end].strip()
            sections.append(Document(page_content=body, metadata={"section": title}))
    else:
        sections.append(Document(page_content=full_text, metadata={"section": "full_document"}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(sections)

    # Noise filter + normalize whitespace; keep full chunk text in metadata as "text"
    def is_noise(doc: Document) -> bool:
        t = (doc.page_content or "").strip().lower()
        if len(t) < 200:
            return True
        toc_keywords = ["appendix", "definitions", "disclosure document", "contact us"]
        if any(k in t for k in toc_keywords) and "purpose:" not in t:
            return True
        if t.count(".") <= 2 and re.search(r"\b\d+\s*$", t):
            return True
        return False

    out_chunks = []
    for doc in chunks:
        if is_noise(doc):
            continue
        text = doc.page_content or ""
        text_norm = re.sub(r"\s*\n+\s*", " ", text)
        text_norm = re.sub(r" {2,}", " ", text_norm).strip()
        meta = dict(doc.metadata or {})
        out_chunks.append(Document(page_content=text_norm, metadata=meta))

    print(f"> Prepared {len(out_chunks)} chunks")
    return out_chunks

# -------------------------
# Embeddings (with retries)
# -------------------------
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
emb = OpenAIEmbeddings(model=EMB_MODEL)

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5), reraise=True)
def embed_batch(batch: List[str]) -> List[List[float]]:
    return emb.embed_documents(batch)

def embed_texts_in_batches(texts: List[str], batch_size: int = BATCH_EMBED) -> List[List[float]]:
    vectors = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        print(f"> Embedding batch {i//batch_size + 1} / {math.ceil(total/batch_size)} (size={len(batch)})")
        vecs = embed_batch(batch)
        vectors.extend(vecs)
        sleep(0.05)
    return vectors

# -------------------------
# Pinecone helpers
# -------------------------
def ensure_index(pc: Pinecone, name: str, dim: int, region: str | None = None):
    idxs = pc.list_indexes()
    if hasattr(idxs, "names"):
        existing = idxs.names()
    elif isinstance(idxs, list):
        existing = idxs
    else:
        existing = []
    if name in existing:
        print(f"> Index '{name}' already exists")
        return
    print(f"> Creating index '{name}' dim={dim}")
    spec = ServerlessSpec(cloud="aws", region=region) if region else None
    pc.create_index(name=name, dimension=dim, metric="cosine", spec=spec)

# -------------------------
# Main ingest flow
# -------------------------
def main():
    chunks = load_and_chunk(PDF_PATH)

    # collect texts, ids, metadata (store full text under 'text' and preview)
    texts, ids, metadatas = [], [], []
    for i, doc in enumerate(chunks):
        txt = doc.page_content or ""
        sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        doc_id = f"chunk-{i}-{sha[:8]}"
        meta = dict(doc.metadata or {})
        meta.setdefault("section", meta.get("section", "unknown"))
        meta["text"] = txt                 # full chunk text (important!)
        meta["text_preview"] = txt[:300]
        meta["sha256"] = sha
        texts.append(txt)
        ids.append(doc_id)
        metadatas.append(meta)

    # get embedding dim
    sample = emb.embed_query("hello")
    dim = len(sample)
    print(f"> embedding dim: {dim}; items: {len(texts)}")

    # embed
    vectors = embed_texts_in_batches(texts, batch_size=BATCH_EMBED)
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding count mismatch")

    # init pinecone and ensure index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, INDEX_NAME, dim=dim, region=PINECONE_REGION)
    index = pc.Index(INDEX_NAME)

    # upsert in batches
    total = len(ids)
    items = [(ids[i], vectors[i], metadatas[i]) for i in range(total)]
    for i in range(0, total, BATCH_UPSERT):
        batch = items[i : i + BATCH_UPSERT]
        print(f"> Upserting batch {i//BATCH_UPSERT + 1} / {math.ceil(total/BATCH_UPSERT)} (size={len(batch)})")
        index.upsert(vectors=batch, namespace=NAMESPACE or None)
        sleep(0.1)

    print("> Done. All chunks upserted.")

if __name__ == "__main__":
    main()

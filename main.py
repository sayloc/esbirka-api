import os
import re
import html
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI

# ---------------------------------------------------------
# UTF-8 JSON OUTPUT (fix pro PowerShell + VSCode)
# ---------------------------------------------------------

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


# ---------------------------------------------------------
# ENV
# ---------------------------------------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL není nastaveno.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY není nastaveno.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------
# DB
# ---------------------------------------------------------

def get_conn():
    return psycopg2.connect(DATABASE_URL)


def run_query(sql: str, params: Optional[tuple] = None) -> List[dict]:
    conn = None
    try:
        conn = get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()
    except Exception as e:
        print("DB error:", repr(e))
        raise
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------
# FASTAPI
# ---------------------------------------------------------

app = FastAPI(
    title="eSbírka Search API",
    default_response_class=UTF8JSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

class SearchResult(BaseModel):
    fragment_id: int
    citace: Optional[str]
    text: str


class RagChunk(BaseModel):
    fragment_id: int
    citation: str
    text: str


class RagResponse(BaseModel):
    chunks: List[RagChunk]


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_text_for_embedding(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = normalize_whitespace(text)
    return text[:4000]


def safe_out(text: str) -> str:
    return html.unescape(text or "").strip()


def embed_query(text: str) -> List[float]:
    clipped = text[:4000] if text else ""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=clipped,
    )
    return resp.data[0].embedding


# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/echo-test")
def echo_test():
    return {"message": "Příliš žluťoučký kůň úpěl ďábelské ódy."}


@app.get("/debug-fragment/{fragment_id}")
def debug_fragment(fragment_id: int):
    sql = """
        SELECT fragment_id, fragment_text
        FROM esb_fragment_text
        WHERE fragment_id = %s
    """
    rows = run_query(sql, (fragment_id,))
    if not rows:
        raise HTTPException(404, "Fragment nenalezen.")

    raw = rows[0]["fragment_text"]
    stripped = clean_text_for_embedding(raw)
    safe = safe_out(stripped)

    return {
        "fragment_id": fragment_id,
        "raw": raw,
        "stripped": stripped,
        "safe": safe
    }


@app.post("/rag-search", response_model=RagResponse)
async def rag_search(request: Request, top_k: int = Query(5, ge=1, le=20)):
    raw = await request.body()

    try:
        contract_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            contract_text = raw.decode("latin-1")
        except Exception:
            contract_text = raw.decode("utf-8", errors="ignore")

    contract_text = clean_text_for_embedding(contract_text)

    if not contract_text:
        return RagResponse(chunks=[])

    try:
        vec = embed_query(contract_text)
    except Exception as e:
        print("Embedding error:", repr(e))
        raise HTTPException(500, "Chyba embeddingu.")

    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    sql = """
        SELECT
            m.fragment_id,
            m.citace_text AS citace,
            t.fragment_text AS text
        FROM esb_fragment_meta m
        JOIN esb_fragment_text t ON t.fragment_id = m.fragment_id
        JOIN esb_fragment_embedding e ON e.fragment_id = m.fragment_id
        WHERE t.fragment_text IS NOT NULL
          AND length(trim(t.fragment_text)) > 20
        ORDER BY e.embedding <-> %s::vector
        LIMIT %s;
    """

    try:
        rows = run_query(sql, (vec_str, top_k))
    except Exception as e:
        print("RAG DB error:", repr(e))
        raise HTTPException(500, "Chyba při RAG dotazu.")

    chunks = []
    for r in rows:
        chunks.append(RagChunk(
            fragment_id=r["fragment_id"],
            citation=safe_out(r.get("citace") or "bez citace"),
            text=normalize_whitespace(safe_out(r.get("text") or "")),
        ))

    return RagResponse(chunks=chunks)

import os
import re
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI

# ---------- ENV ----------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL není nastaveno.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY není nastaveno.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- DB ----------

def get_conn():
    return psycopg2.connect(DATABASE_URL)


def run_query(sql: str, params: Optional[tuple] = None) -> List[dict]:
    conn = None
    try:
        conn = get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
        return rows
    except Exception as e:
        print("DB error:", repr(e))
        raise
    finally:
        if conn is not None:
            conn.close()


# ---------- FastAPI app ----------

app = FastAPI(title="eSbírka Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- MODELY ----------

class SearchResult(BaseModel):
    fragment_id: int
    citace: Optional[str]
    text: str


class RagChunk(BaseModel):
    citation: str
    text: str


class RagResponse(BaseModel):
    chunks: List[RagChunk]


# ---------- UTIL ----------

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_html(text: str) -> str:
    # vyhodí všechny HTML tagy (table, td, p, span…)
    return re.sub(r"<[^>]+>", " ", text or "")


def embed_query(text: str) -> List[float]:
    clipped = text[:4000] if text else ""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=clipped,
    )
    return resp.data[0].embedding


# ---------- ZÁKLADNÍ ENDPOINTY ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/search", response_model=List[SearchResult])
def search(
    query: str = Query(...),
    limit: int = Query(5, ge=1, le=50),
):
    """
    Jednoduchý fulltext přes to_tsvector/plainto_tsquery.
    Používá se hlavně na testování.
    """
    q = query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query nesmí být prázdné.")

    sql = """
        SELECT
            m.fragment_id,
            m.citace_text AS citace,
            t.fragment_text AS text
        FROM esb_fragment_meta m
        JOIN esb_fragment_text t
          ON t.fragment_id = m.fragment_id
        WHERE to_tsvector(
                  'simple',
                  regexp_replace(
                      COALESCE(t.fragment_text, ''),
                      '<[^>]+>', ' ',
                      'g'
                  )
              )
              @@ plainto_tsquery('simple', %s)
        ORDER BY m.fragment_id
        LIMIT %s;
    """

    try:
        rows = run_query(sql, (q, limit))
    except Exception:
        raise HTTPException(status_code=500, detail="Chyba při dotazu do databáze.")

    return rows


# ---------- RAG ENDPOINT (embedding + pgvector) ----------

@app.post("/rag-search", response_model=RagResponse)
async def rag_search(
    request: Request,
    top_k: int = Query(5, ge=1, le=20),
):
    """
    RAG endpoint pro Make:
    - Make pošle syrový text smlouvy v body (text/plain)
    - spočítáme embedding dotazu (OpenAI)
    - v DB najdeme nejbližší fragmenty (pgvector)
    - vrátíme čistý text BEZ HTML tagů
    """

    raw = await request.body()
    contract_text = normalize_whitespace(raw.decode("utf-8"))

    if not contract_text:
        return RagResponse(chunks=[])

    # 1) embedding dotazu
    try:
        vec = embed_query(contract_text)
    except Exception as e:
        print("Embedding error:", repr(e))
        raise HTTPException(status_code=500, detail="Chyba při volání embedding modelu.")

    # převedeme na string ve formátu pgvector
    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    # 2) vektorové hledání v DB
    sql = """
        SELECT
            m.fragment_id,
            m.citace_text AS citace,
            t.fragment_text AS text
        FROM esb_fragment_meta m
        JOIN esb_fragment_text t
          ON t.fragment_id = m.fragment_id
        JOIN esb_fragment_embedding e
          ON e.fragment_id = m.fragment_id
        ORDER BY e.embedding <-> %s::vector
        LIMIT %s;
    """

    try:
        rows = run_query(sql, (vec_str, top_k))
    except Exception as e:
        print("RAG DB error:", repr(e))
        raise HTTPException(status_code=500, detail="Chyba při RAG dotazu do databáze.")

    # 3) očista textu a sestavení odpovědi
    chunks: List[RagChunk] = []
    for r in rows:
        citation = r.get("citace") or "bez citace"
        raw_text = r.get("text") or ""
        clean_text = normalize_whitespace(strip_html(raw_text))
        if not clean_text:
            continue

        chunks.append(
            RagChunk(
                citation=citation,
                text=clean_text,
            )
        )

    return RagResponse(chunks=chunks)

import os
import re
import html
import codecs
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


# ---------- DB připojení ----------

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


# ---------- FastAPI ----------

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


# ---------- POMOCNÉ FUNKCE ----------

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def decode_latin_mess(s: str) -> str:
    """
    Některé texty přijdou jako UTF-8 bajty uložené v Latin-1.
    Tohle se je pokusí převést zpět na normální UTF-8.
    'ÄÃST' -> 'ČÁST'
    """
    if not s:
        return ""
    try:
        return s.encode("latin-1").decode("utf-8")
    except Exception:
        return s


def clean_text_for_embedding(text: str) -> str:
    """
    Vyčistí text před embeddingem:
    - opraví rozbité kódování
    - odstraní HTML tagy
    - znormalizuje mezery
    - ořízne na max 4000 znaků
    """
    if not text:
        return ""
    text = decode_latin_mess(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = normalize_whitespace(text)
    return text[:4000]


def safe_out(text: str) -> str:
    """
    Výstup pro klienta:
    - opraví rozbité kódování
    - dekóduje HTML entity (&aacute; -> á)
    - odstraní HTML tagy
    - znormalizuje mezery
    """
    if not text:
        return ""
    text = decode_latin_mess(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(text)


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

    results: List[SearchResult] = []
    for r in rows:
        results.append(
            SearchResult(
                fragment_id=r["fragment_id"],
                citace=safe_out(r.get("citace")),
                text=safe_out(r.get("text")),
            )
        )
    return results


# ---------- RAG ENDPOINT (embedding RAG) ----------

@app.post("/rag-search", response_model=RagResponse)
async def rag_search(request: Request, top_k: int = Query(5, ge=1, le=20)):
    """
    RAG endpoint pro Make:
    - Make pošle syrový text smlouvy v body (text/plain)
    - Vytvoříme embedding dotazu
    - V DB najdeme nejbližší fragmenty podle pgvector
    """

    raw = await request.body()

    # robustní dekódování requestu
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
        raise HTTPException(
            status_code=500,
            detail="Chyba při volání embedding modelu."
        )

    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

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
        WHERE t.fragment_text IS NOT NULL
          AND length(trim(t.fragment_text)) > 20
        ORDER BY e.embedding <-> %s::vector
        LIMIT %s;
    """

    try:
        rows = run_query(sql, (vec_str, top_k))
    except Exception as e:
        print("RAG DB error:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="Chyba při RAG dotazu do databáze."
        )

    chunks: List[RagChunk] = []
    for r in rows:
        citation = safe_out(r.get("citace") or "bez citace")
        text = safe_out(r.get("text") or "")
        if not text:
            continue
        chunks.append(RagChunk(citation=citation, text=text))

    return RagResponse(chunks=chunks)

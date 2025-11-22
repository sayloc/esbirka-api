import os
import re
import html
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


# ---------- TEXT POMOCNÉ FUNKCE ----------

html_tag_re = re.compile(r"<[^>]+>")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_html(text: str) -> str:
    if not text:
        return ""
    # odstraní HTML tagy
    no_tags = html_tag_re.sub(" ", text)
    # dekóduje HTML entity (&nbsp;, &aacute;)
    unescaped = html.unescape(no_tags)
    return normalize_whitespace(unescaped)


def clean_text_for_embedding(text: str) -> str:
    """
    Čištění vstupního textu (smlouva, dotaz).
    """
    if not text:
        return ""
    return normalize_whitespace(strip_html(text))


def safe_out(text: str) -> str:
    """
    Čištění textu před odesláním ven v JSON (citace i fragment).
    """
    return strip_html(text or "")


def embed_query(text: str) -> List[float]:
    clipped = text[:4000] if text else ""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=clipped,
    )
    return resp.data[0].embedding


# ---------- FastAPI ----------

app = FastAPI(title="eSbírka Search API")

# POZOR: CORSMiddleware se přidává přes add_middleware
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

    # očistíme texty pro výstup
    out: List[SearchResult] = []
    for r in rows:
        out.append(
            SearchResult(
                fragment_id=r["fragment_id"],
                citace=safe_out(r.get("citace")),
                text=safe_out(r.get("text")),
            )
        )
    return out


# ---------- RAG ENDPOINT (embedding RAG) ----------

def decode_body_safe(raw: bytes) -> str:
    """
    Zkusí UTF-8, pak cp1250, pak latin-1.
    Když nic, tak prostě ignoruje chybné bajty.
    """
    for enc in ("utf-8", "cp1250", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


@app.post("/rag-search", response_model=RagResponse)
async def rag_search(request: Request, top_k: int = Query(5, ge=1, le=20)):
    """
    RAG endpoint pro Make:
    - Make pošle syrový text smlouvy v body (text/plain)
    - Vytvoříme embedding dotazu
    - V DB najdeme nejbližší fragmenty podle pgvector
    """

    raw = await request.body()
    if not raw:
        return RagResponse(chunks=[])

    body_text = decode_body_safe(raw)
    contract_text = clean_text_for_embedding(body_text)

    if not contract_text:
        return RagResponse(chunks=[])

    # embedding dotazu
    try:
        vec = embed_query(contract_text)
    except Exception as e:
        print("Embedding error:", repr(e))
        raise HTTPException(status_code=500, detail="Chyba při volání embedding modelu.")

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
          AND length(trim(t.fragment_text)) > 0
        ORDER BY e.embedding <-> %s::vector
        LIMIT %s;
    """

    try:
        rows = run_query(sql, (vec_str, top_k))
    except Exception as e:
        print("RAG DB error:", repr(e))
        raise HTTPException(status_code=500, detail="Chyba při RAG dotazu do databáze.")

    chunks: List[RagChunk] = []
    for r in rows:
        citation = safe_out(r.get("citace") or "bez citace")
        text = safe_out(r.get("text") or "")
        if not text:
            continue
        chunks.append(RagChunk(citation=citation, text=text))

    return RagResponse(chunks=chunks)

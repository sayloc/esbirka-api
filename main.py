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
# UTF-8 JSON OUTPUT
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


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def clean_text_for_embedding(text: str) -> str:
    if not text:
        return ""
    text = strip_html_tags(text)
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


# ---------------------------------------------------------
# /search – čistý fulltext
# ---------------------------------------------------------

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
                text=clean_text_for_embedding(safe_out(r.get("text"))),
            )
        )
    return results


# ---------------------------------------------------------
# /rag-search – HYBRID: fulltext + vektor
# ---------------------------------------------------------

@app.post("/rag-search", response_model=RagResponse)
async def rag_search(request: Request, top_k: int = Query(5, ge=1, le=20)):
    """
    RAG endpoint pro Make:
    - Make pošle syrový text smlouvy v body (text/plain)
    - Vytvoříme embedding dotazu
    - V DB najdeme nejbližší fragmenty podle pgvector,
      ALE jen z těch, které fulltextem odpovídají dotazu.
    """

    raw = await request.body()

    # robustní dekódování
    try:
        contract_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            contract_text = raw.decode("latin-1")
        except Exception:
            contract_text = raw.decode("utf-8", errors="ignore")

    # text pro embedding
    cleaned_for_embed = clean_text_for_embedding(contract_text)
    if not cleaned_for_embed:
        return RagResponse(chunks=[])

    # text pro fulltext (může být stejný)
    fulltext_query = cleaned_for_embed

    # embedding dotazu
    try:
        vec = embed_query(cleaned_for_embed)
    except Exception as e:
        print("Embedding error:", repr(e))
        raise HTTPException(500, "Chyba embeddingu.")

    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    # HYBRID: fulltext + vektor
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
          AND length(trim(t.fragment_text)) > 50
          AND to_tsvector(
                'simple',
                regexp_replace(
                    COALESCE(t.fragment_text, ''),
                    '<[^>]+>', ' ',
                    'g'
                )
              )
              @@ plainto_tsquery('simple', %s)
        ORDER BY e.embedding <-> %s::vector
        LIMIT %s;
    """

    try:
        rows = run_query(sql, (fulltext_query, vec_str, top_k))
    except Exception as e:
        print("RAG DB error:", repr(e))
        raise HTTPException(500, "Chyba při RAG dotazu.")

    chunks: List[RagChunk] = []
    for r in rows:
        citation = safe_out(r.get("citace") or "bez citace")
        # tady už HTML TAGY STRHÁVÁME
        text = clean_text_for_embedding(safe_out(r.get("text") or ""))

        if not text:
            continue

        chunks.append(RagChunk(
            fragment_id=r["fragment_id"],
            citation=citation,
            text=text,
        ))

    return RagResponse(chunks=chunks)

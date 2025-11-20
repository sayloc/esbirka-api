import os
import re
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor


# ---------- DB připojení ----------

def get_conn():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL není nastaveno.")
    return psycopg2.connect(db_url)


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


# ---------- ENDPOINTY ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    # pro Render health check
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

    return rows


# ---------- RAG ENDPOINT PRO MAKE (vrací reálné paragrafy) ----------

def _build_query_from_contract(text: str, max_words: int = 20) -> str:
    """
    Vytáhne z textu smlouvy prvních ~max_words „normálních“ slov
    a z nich udělá jednoduchý fulltext dotaz.
    """
    # povolíme česká písmena a čísla
    words = re.findall(r"[0-9A-Za-zÁ-Žá-ž]+", text)
    if not words:
        return ""
    # vezmeme jen prvních N slov, ať tsquery není mega dlouhý
    return " ".join(words[:max_words])


@app.post("/rag-search", response_model=RagResponse)
async def rag_search(request: Request, top_k: int = Query(5, ge=1, le=20)):
    """
    RAG endpoint volaný z Make.com:
    - Make pošle syrový text smlouvy v body (text/plain).
    - My z něj uděláme jednoduchý fulltext dotaz a vrátíme
      několik relevantních fragmentů zákonů.
    """

    raw = await request.body()
    contract_text = raw.decode("utf-8").strip()

    if not contract_text:
        return RagResponse(chunks=[])

    q = _build_query_from_contract(contract_text)
    if not q:
        return RagResponse(chunks=[])

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
        rows = run_query(sql, (q, top_k))
    except Exception:
        raise HTTPException(status_code=500, detail="Chyba při RAG dotazu do databáze.")

    chunks: List[RagChunk] = []
    for r in rows:
        citation = r.get("citace") or ""
        text = r.get("text") or ""
        if not text:
            continue
        chunks.append(
            RagChunk(
                citation=citation or "bez citace",
                text=text,
            )
        )

    return RagResponse(chunks=chunks)


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
    """
    Obecná helper funkce pro SELECTy.
    Když něco spadne, vytiskneme SQL + params do logu.
    """
    conn = None
    try:
        conn = get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # DEBUG
            # print("SQL:", sql)
            # print("PARAMS:", params)

            if params is not None:
                cur.execute(sql, params)
            else:
                cur.execute(sql)

            rows = cur.fetchall()
        return rows
    except Exception as e:
        print("DB error:", repr(e))
        print("SQL used:", sql)
        print("PARAMS used:", params)
        raise
    finally:
        if conn is not None:
            conn.close()


# ---------- FastAPI ----------

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


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


# ---------- MODELY ----------

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


# ---------- POMOCNÉ FUNKCE ----------

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_html(text: str) -> str:
    if not text:
        return ""
    # pryč <tagy>
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(text)


def clean_text_for_embedding(text: str) -> str:
    """
    Vyčistí text před embeddingem:
    - odstraní HTML tagy
    - znormalizuje whitespace
    - ořeže na ~4000 znaků
    """
    text = strip_html(text)
    return text[:4000]


def safe_out(text: str) -> str:
    """
    Výstup pro klienta:
    - dekóduje HTML entity (&aacute; -> á)
    - ořízne mezery
    """
    return html.unescape(text or "").strip()


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


@app.get("/echo-test")
def echo_test():
    # rychlý test UTF-8
    return {"message": "Příliš žluťoučký kůň úpěl ďábelské ódy."}


@app.get("/debug-fragment/{fragment_id}")
def debug_fragment(fragment_id: int):
    """
    Debug: načte syrový text fragmentu a ukáže raw/stripped/safe.
    """
    sql = """
        SELECT fragment_text
        FROM esb_fragment_text
        WHERE fragment_id = %s;
    """
    rows = run_query(sql, (fragment_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Fragment nenalezen.")

    raw = rows[0]["fragment_text"] or ""
    stripped = strip_html(raw)
    safe = safe_out(stripped)

    return {
        "fragment_id": fragment_id,
        "raw": raw,
        "stripped": stripped,
        "safe": safe,
    }


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
                text=normalize_whitespace(safe_out(r.get("text"))),
            )
        )
    return results


# ---------- RAG ENDPOINT (embedding RAG) ----------

@app.post("/rag-search", response_model=RagResponse)
async def rag_search(request: Request, top_k: int = Query(5, ge=1, le=20)):
    """
    RAG endpoint:
    - klient pošle syrový text smlouvy / dotazu v body (text/plain)
    - vytvoříme embedding dotazu
    - v DB najdeme nejbližší fragmenty (pgvector)
    """

    raw = await request.body()

    # robustní dekódování vstupu
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

    # --- EMBEDDING DOTAZU ---
    try:
        vec = embed_query(contract_text)
    except Exception as e:
        print("Embedding error:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Embedding error: {repr(e)}",
        )

    # složíme pgvector literál: '[0.123,0.456,...]'
    vec_literal = "[" + ",".join(str(x) for x in vec) + "]"

    # --- RAG DOTAZ DO DB ---
    sql = f"""
        SELECT
            e.fragment_id,
            m.citace_text AS citace,
            t.fragment_text AS text
        FROM esb_fragment_embedding e
        JOIN esb_fragment_meta m
          ON m.fragment_id = e.fragment_id
        JOIN esb_fragment_text t
          ON t.fragment_id = e.fragment_id
        WHERE t.fragment_text IS NOT NULL
          AND length(trim(t.fragment_text)) > 50
          AND m.citace_text LIKE '§ %'
        ORDER BY e.embedding <-> '{vec_literal}'::vector
        LIMIT {top_k};
    """

    try:
        rows = run_query(sql)
    except Exception as e:
        print("RAG DB error:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"RAG DB error: {repr(e)}",
        )

    chunks: List[RagChunk] = []
    for r in rows:
        raw_text = r.get("text") or ""
        cleaned_text = clean_text_for_embedding(raw_text)
        if not cleaned_text:
            continue

        citation = safe_out(r.get("citace") or "bez citace")

        chunks.append(
            RagChunk(
                fragment_id=r["fragment_id"],
                citation=citation,
                text=cleaned_text,
            )
        )

    return RagResponse(chunks=chunks)

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# ---------- DB připojení ----------

def get_conn():
    """
    Vytáhne DATABASE_URL z .env nebo z env proměnných a vrátí psycopg2 connection.
    """
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL není nastaveno (v .env nebo v prostředí).")
    return psycopg2.connect(db_url)


def run_query(sql: str, params: Optional[tuple] = None) -> List[dict]:
    """
    Spustí SQL dotaz a vrátí list dictů (RealDictCursor).
    V případě chyby jen zaloguje a vyhodí výjimku dál.
    """
    conn = None
    try:
        conn = get_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
        return rows
    except Exception as e:
        # důležitý log, ať vidíme skutečnou chybu v konzoli
        print("DB error in /search:", repr(e))
        raise
    finally:
        if conn is not None:
            conn.close()


# ---------- FastAPI app ----------

app = FastAPI(title="eSbírka Search API")

# CORS – klidně si přitvrď podle potřeby (např. jen tvůj Bubble domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # pro testování necháme všude
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    fragment_id: int
    citace: Optional[str]
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search", response_model=List[SearchResult])
def search(
    query: str = Query(..., description="Fulltext dotaz (česky)"),
    limit: int = Query(5, ge=1, le=50, description="Maximální počet výsledků"),
):
    """
    Jednoduchý fulltext nad esb_fragment_meta + esb_fragment_text.
    - spočítá to_tsvector('simple', ...) in-place
    - hledá pomocí plainto_tsquery('simple', :query)
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
        # už jsme zalogovali v run_query, tady jen pošleme generickou chybu ven
        raise HTTPException(status_code=500, detail="Chyba při dotazu do databáze.")

    # Pydantic si to přemapuje podle klíčů
    return rows

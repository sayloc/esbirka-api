import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Načíst .env a DB URL
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL není v .env")


def get_conn():
    # Jednoduché připojení pro každý request
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)


app = FastAPI(title="eSbírka search API")

# CORS – ať se na to může připojit Bubble / web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    fragment_id: int
    citace: str | None = None
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search", response_model=list[SearchResult])
def search(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
):
    q = query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Dotaz nesmí být prázdný.")

    # Hledáme v tabulce esb_fragment_text, kde je FTS sloupec `fts`
    # a připojíme si meta kvůli citaci.
    sql = """
        SELECT
            m.fragment_id,
            m.citace,
            t.fragment_text AS text
        FROM esb_fragment_text t
        JOIN esb_fragment_meta m
          ON m.fragment_id = t.fragment_id
        WHERE t.fts @@ plainto_tsquery('simple', %s)
        ORDER BY ts_rank(t.fts, plainto_tsquery('simple', %s)) DESC
        LIMIT %s;
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (q, q, limit))
                rows = cur.fetchall()
    except Exception as e:
        print("DB error in /search:", repr(e))
        raise HTTPException(status_code=500, detail="Chyba při dotazu do databáze.")

    results: list[SearchResult] = []
    for row in rows:
        results.append(
            SearchResult(
                fragment_id=row["fragment_id"],
                citace=row.get("citace"),
                text=row["text"],
            )
        )

    return results


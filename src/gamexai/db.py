"""Database layer (SQLAlchemy) for persisting everything the pipeline produces.

Engine-agnostic: set DATABASE_URL (in .env) to a PostgreSQL URL, e.g.
    postgresql+psycopg2://postgres:PASSWORD@localhost:5432/gamexai
If DATABASE_URL is unset, or the server can't be reached, it falls back to a
local SQLite file (results/gamexai.db) so a demo never breaks.

For PostgreSQL the target database is created automatically if it doesn't exist.
"""
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from . import config as C

SQLITE_URL = f"sqlite:///{(C.RESULTS / 'gamexai.db').as_posix()}"


def _ensure_postgres_db(url):
    """Create the target PostgreSQL database if it is missing."""
    admin = url.set(database="postgres")
    eng = create_engine(admin, isolation_level="AUTOCOMMIT")
    with eng.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"),
            {"n": url.database}).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    eng.dispose()


def get_engine(echo=False):
    """Return a working SQLAlchemy engine, falling back to SQLite on any failure."""
    raw = os.getenv("DATABASE_URL")
    if not raw:
        return create_engine(SQLITE_URL, echo=echo), "sqlite (default)"
    try:
        url = make_url(raw)
        if url.get_backend_name() == "postgresql":
            _ensure_postgres_db(url)
        eng = create_engine(url, echo=echo)
        with eng.connect() as conn:                       # smoke-test the connection
            conn.execute(text("SELECT 1"))
        return eng, f"{url.get_backend_name()}:{url.database}"
    except Exception as e:                                 # unreachable -> SQLite
        print(f"[db] '{raw.split('@')[-1] if '@' in raw else raw}' unavailable "
              f"({type(e).__name__}); falling back to SQLite.")
        return create_engine(SQLITE_URL, echo=echo), "sqlite (fallback)"

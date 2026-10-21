"""Database layer (SQLAlchemy) for persisting everything the pipeline produces.

Engine-agnostic: set DATABASE_URL (in .env) to any SQLAlchemy URL, e.g.
    PostgreSQL:  postgresql+psycopg2://postgres:PASSWORD@localhost:5432/gamexai
    SQL Server:  mssql+pyodbc://@localhost/gamexai?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes
If DATABASE_URL is unset, or the server can't be reached, it falls back to a
local SQLite file (results/gamexai.db) so a demo never breaks.

For PostgreSQL and SQL Server the target database is created automatically.
"""
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from . import config as C

SQLITE_URL = f"sqlite:///{(C.RESULTS / 'gamexai.db').as_posix()}"


def _ensure_database(url):
    """Create the target database if it is missing (PostgreSQL / SQL Server)."""
    backend = url.get_backend_name()
    admin_db = {"postgresql": "postgres", "mssql": "master"}.get(backend)
    if not admin_db:
        return
    eng = create_engine(url.set(database=admin_db), isolation_level="AUTOCOMMIT")
    with eng.connect() as conn:
        if backend == "postgresql":
            exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname=:n"),
                                  {"n": url.database}).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
        else:  # mssql
            exists = conn.execute(text("SELECT 1 FROM sys.databases WHERE name=:n"),
                                  {"n": url.database}).scalar()
            if not exists:
                conn.execute(text(f"CREATE DATABASE [{url.database}]"))
    eng.dispose()


def get_engine(echo=False):
    """Return a working SQLAlchemy engine, falling back to SQLite on any failure."""
    raw = os.getenv("DATABASE_URL")
    if not raw:
        return create_engine(SQLITE_URL, echo=echo), "sqlite (default)"
    try:
        url = make_url(raw)
        _ensure_database(url)
        kwargs = {"echo": echo}
        if url.get_backend_name() == "mssql":
            kwargs["fast_executemany"] = True             # fast bulk inserts
        eng = create_engine(url, **kwargs)
        with eng.connect() as conn:                       # smoke-test the connection
            conn.execute(text("SELECT 1"))
        return eng, f"{url.get_backend_name()}:{url.database}"
    except Exception as e:                                 # unreachable -> SQLite
        print(f"[db] '{raw.split('@')[-1] if '@' in raw else raw}' unavailable "
              f"({type(e).__name__}); falling back to SQLite.")
        return create_engine(SQLITE_URL, echo=echo), "sqlite (fallback)"

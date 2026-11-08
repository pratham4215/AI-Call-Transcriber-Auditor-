"""
db.py — read-only PostgreSQL access for CRM / call data.

Connection settings come from config.py (DATABASE_URL, or DB_HOST / DB_NAME /
DB_USER / DB_PASSWORD in .env). Every transaction is opened READ ONLY with a
statement timeout, so the audit app can neither modify the CRM database nor
hang on a slow query.

Check the connection and list the tables the DB user can see:
    python -m core.db
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List, Optional

from config import (
    DATABASE_URL, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE,
    DB_CONNECT_TIMEOUT, DB_STATEMENT_TIMEOUT_MS, IMPORT_WORKERS,
)

try:
    from sqlalchemy import create_engine, event, text
    from sqlalchemy.engine import URL, Engine, make_url
except ImportError:  # pragma: no cover - DB support is optional
    create_engine = None  # type: ignore

_ENGINE: Optional["Engine"] = None
_ENGINE_LOCK = threading.Lock()
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def is_configured() -> bool:
    return bool(DATABASE_URL or (DB_HOST and DB_NAME and DB_USER))


def _url() -> "URL":
    if DATABASE_URL:
        url = make_url(DATABASE_URL)
        if url.drivername in {"postgres", "postgresql"}:
            url = url.set(drivername="postgresql+psycopg")
        return url
    # URL.create escapes special characters in the password; never build it by hand.
    return URL.create(
        "postgresql+psycopg",
        username=DB_USER, password=DB_PASSWORD or None,
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
    )


def get_engine() -> "Engine":
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    if create_engine is None:
        raise RuntimeError('PostgreSQL support needs: pip install sqlalchemy "psycopg[binary]"')
    if not is_configured():
        raise RuntimeError("Database not configured: set DB_HOST, DB_NAME, DB_USER and DB_PASSWORD in .env")
    with _ENGINE_LOCK:
        if _ENGINE is None:
            connect_args: Dict[str, Any] = {
                "connect_timeout": DB_CONNECT_TIMEOUT,
                "application_name": "kataria-audit",
            }
            if DB_SSLMODE:
                connect_args["sslmode"] = DB_SSLMODE
            engine = create_engine(
                _url(),
                connect_args=connect_args,
                # BEGIN READ ONLY on every transaction (also works through PgBouncer)
                execution_options={"postgresql_readonly": True},
                # Every import worker can hold a connection, with 10 more for the dashboard's own requests.
                pool_size=10, max_overflow=IMPORT_WORKERS, pool_pre_ping=True, pool_recycle=1800,
            )

            @event.listens_for(engine, "connect")
            def _session_settings(dbapi_connection, _record):
                # Once per connection instead of per query: one round trip less over the VPN.
                with dbapi_connection.cursor() as cursor:
                    cursor.execute(f"SET statement_timeout = {int(DB_STATEMENT_TIMEOUT_MS)}")
                dbapi_connection.commit()

            _ENGINE = engine
    return _ENGINE


def fetch_all(sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Run one read-only query and return its rows as dicts."""
    with get_engine().connect() as conn:
        return [dict(row) for row in conn.execute(text(sql), params or {}).mappings()]


def quote_table(name: str) -> str:
    """Quote a configured 'table' or 'schema.table' name for use in SQL."""
    parts = name.split(".")
    if not 1 <= len(parts) <= 2 or not all(_IDENT_RE.match(part) for part in parts):
        raise ValueError(f"Invalid table name: {name!r}")
    preparer = get_engine().dialect.identifier_preparer
    return ".".join(preparer.quote(part) for part in parts)


def describe_error(exc: BaseException) -> str:
    """One-line driver error (no SQL, no credentials) for logs and API responses."""
    orig = getattr(exc, "orig", None) or exc
    lines = str(orig).strip().splitlines()
    return f"{type(orig).__name__}: {lines[0] if lines else ''}".rstrip(": ")


def health() -> Dict[str, Any]:
    """Connection status for /api/db/health; never includes credentials."""
    info: Dict[str, Any] = {"configured": is_configured(), "connected": False}
    if not info["configured"]:
        return info
    started = time.perf_counter()
    try:
        row = fetch_all(
            "SELECT current_database() AS database, current_setting('server_version') AS server_version, "
            "current_setting('transaction_read_only') AS read_only"
        )[0]
    except Exception as exc:
        info["error"] = describe_error(exc)
        return info
    info.update(
        connected=True,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        database=row["database"],
        server_version=row["server_version"],
        read_only=row["read_only"] == "on",
    )
    return info


def describe_tables(keyword: str = "") -> List[Dict[str, Any]]:
    """Tables and views visible to the DB user, with approximate row counts and columns.

    keyword keeps only tables whose schema, name or a column name contains it.
    """
    return fetch_all("""
        SELECT * FROM (
            SELECT t.table_schema, t.table_name, t.table_type,
                   NULLIF(pc.reltuples, -1)::bigint AS approx_rows,
                   (SELECT string_agg(c.column_name || ' ' || c.data_type, ', ' ORDER BY c.ordinal_position)
                    FROM information_schema.columns c
                    WHERE c.table_schema = t.table_schema AND c.table_name = t.table_name) AS columns
            FROM information_schema.tables t
            LEFT JOIN pg_namespace n ON n.nspname = t.table_schema
            LEFT JOIN pg_class pc ON pc.relnamespace = n.oid AND pc.relname = t.table_name
            WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
        ) tables
        WHERE :kw = '' OR (table_schema || '.' || table_name || ' ' || coalesce(columns, '')) ILIKE '%' || :kw || '%'
        ORDER BY table_schema, table_name
    """, {"kw": keyword})


if __name__ == "__main__":
    import sys

    status = health()
    print(status)
    if status.get("connected"):
        tables = describe_tables(sys.argv[1] if len(sys.argv) > 1 else "")
        for table in tables:
            rows = "?" if table["approx_rows"] is None else f"~{table['approx_rows']:,}"
            print(f"\n{table['table_schema']}.{table['table_name']} ({table['table_type']}, {rows} rows)")
            print(f"  {table['columns']}")
        print(f"\n{len(tables)} tables/views")

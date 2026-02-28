"""
Database management utilities for MCP Database Server (QA edition).

Handles connection management, query execution, and safety checks.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    import aiosqlite
except ImportError:
    aiosqlite = None

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)


def make_url(schema_id: str) -> str:
    """Build a fully-qualified DB URL from BASE_URL + schema_id.

    BASE_URL must be set to a URL with a trailing slash, e.g.:
        mysql+aiomysql://user:pass@host.docker.internal:3307/
    """
    base = os.environ["BASE_URL"]
    return base.rstrip("/") + "/" + schema_id


class DatabaseManager:
    """Manages an async database connection for a single schema."""

    def __init__(self, url: str):
        self.database_url = url
        self.database_type = self._detect_database_type()
        self.engine = None
        self._initialize_engine()

    def _detect_database_type(self) -> str:
        if "postgresql" in self.database_url or "postgres" in self.database_url:
            return "postgresql"
        elif "mysql" in self.database_url:
            return "mysql"
        elif "sqlite" in self.database_url:
            return "sqlite"
        return "postgresql"

    def _initialize_engine(self):
        try:
            connect_args = {}
            if self.database_type == "mysql":
                url_lower = self.database_url.lower()
                if (
                    "ssl_mode=required" in url_lower
                    or "ssl-mode=required" in url_lower
                    or os.getenv("MYSQL_SSL", "false").lower() == "true"
                ):
                    import ssl
                    ssl_context = ssl.create_default_context()
                    connect_args["ssl"] = ssl_context
                    logger.info("MySQL SSL context enabled.")
            self.engine = create_async_engine(
                self.database_url,
                poolclass=NullPool,
                echo=False,
                connect_args=connect_args,
            )
            logger.info(f"Database engine initialized for {self.database_type}")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    async def test_connection(self) -> bool:
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def list_tables(
        self, filter: str = "", include_row_estimates: bool = False
    ) -> List[Dict[str, Any]]:
        """List tables in the current schema.

        Args:
            filter: Substring filter applied to table_name (LIKE '%filter%').
            include_row_estimates: Include TABLE_ROWS estimate (MySQL only; approximate).
        """
        try:
            async with self.engine.begin() as conn:
                params: dict = {}

                if self.database_type == "postgresql":
                    filter_sql = ""
                    if filter:
                        filter_sql = "AND t.table_name LIKE :filter"
                        params["filter"] = f"%{filter}%"
                    query = text(f"""
                        SELECT
                            t.table_name,
                            COUNT(c.column_name) AS column_count
                        FROM information_schema.tables t
                        LEFT JOIN information_schema.columns c
                          ON c.table_schema = t.table_schema
                         AND c.table_name = t.table_name
                        WHERE t.table_schema = 'public'
                          AND t.table_type = 'BASE TABLE'
                          {filter_sql}
                        GROUP BY t.table_name
                        ORDER BY t.table_name
                    """)

                elif self.database_type == "sqlite":
                    filter_sql = ""
                    if filter:
                        filter_sql = "AND name LIKE :filter"
                        params["filter"] = f"%{filter}%"
                    query = text(f"""
                        SELECT name AS table_name, 0 AS column_count
                        FROM sqlite_master
                        WHERE type = 'table'
                          AND name NOT LIKE 'sqlite_%'
                          {filter_sql}
                        ORDER BY name
                    """)

                else:  # MySQL
                    filter_sql = ""
                    if filter:
                        filter_sql = "AND t.table_name LIKE :filter"
                        params["filter"] = f"%{filter}%"
                    row_est_col = ", t.table_rows AS row_estimate" if include_row_estimates else ""
                    row_est_group = ", t.table_rows" if include_row_estimates else ""
                    query = text(f"""
                        SELECT
                            t.table_name,
                            COUNT(c.column_name) AS column_count
                            {row_est_col}
                        FROM information_schema.tables t
                        LEFT JOIN information_schema.columns c
                          ON c.table_schema = t.table_schema
                         AND c.table_name = t.table_name
                        WHERE t.table_schema = DATABASE()
                          AND t.table_type = 'BASE TABLE'
                          AND t.table_schema NOT IN
                              ('information_schema', 'performance_schema', 'sys', 'mysql')
                          {filter_sql}
                        GROUP BY t.table_name {row_est_group}
                        ORDER BY t.table_name
                    """)

                result = await conn.execute(query, params)
                tables = []
                for row in result:
                    table_info: Dict[str, Any] = {
                        "table_name": row[0],
                        "column_count": row[1] if len(row) > 1 else 0,
                    }
                    if self.database_type == "sqlite" and table_info["column_count"] == 0:
                        col_result = await conn.execute(
                            text(f"PRAGMA table_info({row[0]})")
                        )
                        table_info["column_count"] = len(list(col_result))
                    if include_row_estimates and self.database_type == "mysql" and len(row) > 2:
                        table_info["row_estimate"] = row[2] or 0
                    tables.append(table_info)
                return tables
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            raise

    async def describe_table(
        self,
        table_name: str,
        include_indexes: bool = True,
        include_fks: bool = True,
    ) -> Dict[str, Any]:
        """Get column info, index info, and FK info for a table.

        Returns:
            {
              "columns": [{"column_name", "data_type", "is_nullable"}, ...],
              "indexes": {index_name: {"columns": [...], "unique": bool}, ...},
              "foreign_keys": [{"column", "references_table", "references_column",
                                "constraint_name"}, ...],
            }

        Note: index and FK info is MySQL-only; returns empty dicts/lists for other DBs.
        """
        try:
            async with self.engine.begin() as conn:
                # --- columns ---
                if self.database_type == "postgresql":
                    result = await conn.execute(text("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = :t
                        ORDER BY ordinal_position
                    """), {"t": table_name})
                    columns = [
                        {"column_name": r[0], "data_type": r[1], "is_nullable": r[2] == "YES"}
                        for r in result
                    ]
                elif self.database_type == "sqlite":
                    result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
                    columns = [
                        {"column_name": r[1], "data_type": r[2], "is_nullable": not bool(r[3])}
                        for r in result
                    ]
                else:  # MySQL
                    result = await conn.execute(text("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE()
                          AND table_name = :t
                        ORDER BY ordinal_position
                    """), {"t": table_name})
                    columns = [
                        {"column_name": r[0], "data_type": r[1], "is_nullable": r[2] == "YES"}
                        for r in result
                    ]

                out: Dict[str, Any] = {
                    "columns": columns,
                    "indexes": {},
                    "foreign_keys": [],
                }

                # --- indexes (MySQL only) ---
                if include_indexes and self.database_type == "mysql":
                    idx_result = await conn.execute(
                        text(f"SHOW INDEX FROM `{table_name}`")
                    )
                    idx_map: Dict[str, Any] = {}
                    for r in idx_result:
                        idx_name = r[2]    # Key_name
                        col_name = r[4]    # Column_name
                        non_unique = r[1]  # Non_unique
                        if idx_name not in idx_map:
                            idx_map[idx_name] = {
                                "columns": [],
                                "unique": not bool(non_unique),
                            }
                        idx_map[idx_name]["columns"].append(col_name)
                    out["indexes"] = idx_map

                # --- foreign keys (MySQL only) ---
                if include_fks and self.database_type == "mysql":
                    schema_row = await conn.execute(text("SELECT DATABASE()"))
                    schema_name = schema_row.scalar()
                    fk_result = await conn.execute(text("""
                        SELECT column_name, referenced_table_name,
                               referenced_column_name, constraint_name
                        FROM information_schema.key_column_usage
                        WHERE table_schema = :schema
                          AND table_name = :t
                          AND referenced_table_name IS NOT NULL
                        ORDER BY constraint_name, ordinal_position
                    """), {"schema": schema_name, "t": table_name})
                    out["foreign_keys"] = [
                        {
                            "column": r[0],
                            "references_table": r[1],
                            "references_column": r[2],
                            "constraint_name": r[3],
                        }
                        for r in fk_result
                    ]

                return out
        except Exception as e:
            logger.error(f"Error describing table {table_name}: {e}")
            raise

    async def execute_safe_query(
        self,
        query: str,
        params: dict = None,
        limit: int = 200,
        timeout_ms: int = 3000,
    ) -> List[Dict[str, Any]]:
        """Execute a read-only SQL query with strict safety enforcement.

        Enforces:
          - SELECT/WITH prefix only
          - Single statement (no bare semicolons outside string literals)
          - Denylist of mutating/dangerous keywords
          - INTO OUTFILE/DUMPFILE block
          - MAX_EXECUTION_TIME optimizer hint (MySQL)
          - LIMIT injection/replacement
        """
        # 1. Strip SQL comments
        cleaned = re.sub(r'--[^\n]*', '', query)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # 2. First token must be SELECT or WITH
        first_token = re.split(r'\s+', cleaned, maxsplit=1)[0].upper() if cleaned else ""
        if first_token not in ("SELECT", "WITH"):
            raise ValueError("Only SELECT (or WITH … SELECT) queries are allowed.")

        # 3. Single-statement: strip string literals, then reject bare semicolons
        no_strings = re.sub(r"'(?:[^'\\]|\\.)*'", "''", cleaned)
        no_strings = re.sub(r'"(?:[^"\\]|\\.)*"', '""', no_strings)
        if ';' in no_strings:
            raise ValueError("Multi-statement queries are not allowed.")

        # 4. Denylist check on full token stream (word-boundary match)
        _DENYLIST = frozenset({
            'INSERT', 'UPDATE', 'DELETE', 'REPLACE', 'MERGE',
            'CREATE', 'ALTER', 'DROP', 'TRUNCATE',
            'GRANT', 'REVOKE', 'SET', 'LOCK', 'UNLOCK', 'KILL',
            'CALL', 'DO', 'PREPARE', 'EXECUTE', 'DEALLOCATE',
            'LOAD', 'OUTFILE', 'INFILE',
        })
        tokens = set(re.findall(r'\b[A-Za-z_]+\b', no_strings.upper()))
        bad = tokens & _DENYLIST
        if bad:
            raise ValueError(
                f"Query contains disallowed keyword(s): {', '.join(sorted(bad))}"
            )

        # 5. Block INTO OUTFILE / INTO DUMPFILE
        if re.search(r'\bINTO\s+(?:OUTFILE|DUMPFILE)\b', cleaned, re.IGNORECASE):
            raise ValueError("INTO OUTFILE/DUMPFILE is not allowed.")

        # 6. Inject MAX_EXECUTION_TIME optimizer hint (MySQL only)
        if self.database_type == "mysql":
            cleaned = re.sub(
                r'^(SELECT)\b',
                f'SELECT /*+ MAX_EXECUTION_TIME({timeout_ms}) */',
                cleaned,
                count=1,
                flags=re.IGNORECASE,
            )

        # 7. Inject or replace LIMIT
        if re.search(r'\bLIMIT\b', cleaned, re.IGNORECASE):
            cleaned = re.sub(
                r'\bLIMIT\s+\d+', f'LIMIT {limit}', cleaned, flags=re.IGNORECASE
            )
        else:
            cleaned = f"{cleaned.rstrip(';')} LIMIT {limit}"

        # 8. Execute with named bind parameters
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text(cleaned), params or {})
                rows = []
                for row in result:
                    row_dict: Dict[str, Any] = {}
                    for i, col in enumerate(result.keys()):
                        value = row[i]
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        elif not isinstance(value, (str, int, float, bool, type(None))):
                            value = str(value)
                        row_dict[col] = value
                    rows.append(row_dict)
                return rows
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise


# ---------------------------------------------------------------------------
# FastAPI dependency — debug HTTP server (app/server.py) only
# Not used by mcp_server.py.
# ---------------------------------------------------------------------------

_db_manager: Optional[DatabaseManager] = None


async def get_db_manager() -> DatabaseManager:
    """Create/reuse a DatabaseManager for the debug HTTP server profile.

    Reads DATABASE_URL (or DB_URL/POSTGRES_URL/MYSQL_URL) from environment.
    Not used by mcp_server.py — only app/server.py (debug profile).
    """
    global _db_manager
    if _db_manager is None:
        db_url = (
            os.getenv("DATABASE_URL")
            or os.getenv("DB_URL")
            or os.getenv("POSTGRES_URL")
            or os.getenv("MYSQL_URL")
        )
        if not db_url:
            raise Exception("DATABASE_URL env var not set")
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif db_url.startswith("mysql://"):
            db_url = db_url.replace("mysql://", "mysql+aiomysql://", 1)
        _db_manager = DatabaseManager(db_url)
        if not await _db_manager.test_connection():
            raise Exception("Cannot connect to database")
    return _db_manager


async def cleanup_db_manager():
    """Cleanup database manager"""
    global _db_manager
    if _db_manager and _db_manager.engine:
        await _db_manager.engine.dispose()
        _db_manager = None

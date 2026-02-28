#!/usr/bin/env python3
"""
QA MCP Server — DB tools + Grails symbol index.

Two sets of tools:
  1. DB tools (6): multi-schema MySQL navigation against QA1 environment
  2. Symbol index tools (3 + reload): ported from mcp-indexed-tabs/symbol_index_mcp.py

Transport:
  --transport stdio   stdio MCP (default; for direct integration)
  --transport http    streamable-HTTP on port 8000 (Docker default)

Claude Code config for HTTP transport:
  { "mcpServers": { "qa-mcp": { "url": "http://localhost:8000/mcp" } } }
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add app/ to Python path — works both locally and in Docker (/app/app)
_script_dir = Path(__file__).resolve().parent
_app_dir = _script_dir / "app"
if _app_dir.exists():
    sys.path.insert(0, str(_app_dir))
else:
    sys.path.insert(0, "/app/app")

from mcp.server import fastmcp
from dotenv import load_dotenv
from db import DatabaseManager, make_url

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qa-mcp-server")

# ---------------------------------------------------------------------------
# FastMCP instance
# host/port are used when running in streamable-HTTP mode.
# The MCP endpoint will be at http://localhost:8000/mcp
# ---------------------------------------------------------------------------
mcp = fastmcp.FastMCP("QA DB & Symbol Server", host="0.0.0.0", port=8000)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_data_sources: List[Dict[str, Any]] = []   # allowlist built at startup
_active_schema_id: str = "mag"             # default; changed by use_data_source
_managers: Dict[str, DatabaseManager] = {} # lazily-populated cache

_symbol_db_path: Optional[str] = None     # None = symbol tools disabled

# Display-name column in oasis.oem.
# CONFIRM against actual schema before deploying — common candidates: "name", "display_name".
# If wrong, initialize_data_sources() will log a warning and skip tenant discovery.
_OEM_DISPLAY_COL = "name"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_active_manager() -> DatabaseManager:
    """Return (or lazily create) the DatabaseManager for the active schema."""
    if _active_schema_id not in _managers:
        _managers[_active_schema_id] = DatabaseManager(make_url(_active_schema_id))
    return _managers[_active_schema_id]


def _format_ann_attr(ann_name: str, attrs_json: Optional[str]) -> str:
    """Format a method annotation as [@AnnName] or [@AnnName(attrs)]."""
    if not attrs_json:
        return f"[@{ann_name}]"
    try:
        attrs = json.loads(attrs_json)
    except Exception:
        return f"[@{ann_name}]"
    if ann_name == "Secured" and "roles" in attrs:
        return f"[@{ann_name}({', '.join(attrs['roles'])})]"
    if "raw" in attrs:
        return f"[@{ann_name}({attrs['raw'][:40]})]"
    return f"[@{ann_name}]"


def _sym_conn() -> sqlite3.Connection:
    """Open a fresh sqlite3 connection to the symbol index."""
    con = sqlite3.connect(_symbol_db_path)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

async def initialize_data_sources():
    """Discover and register all available data sources at startup."""
    global _data_sources, _managers

    base_url = os.environ.get("BASE_URL")
    if not base_url:
        raise ValueError("BASE_URL environment variable is required.")

    # Use mag manager for all discovery queries
    mag_mgr = DatabaseManager(make_url("mag"))
    if not await mag_mgr.test_connection():
        raise RuntimeError(
            "Cannot connect to mag schema — check BASE_URL credentials and network."
        )

    async def schema_exists(schema_name: str) -> bool:
        rows = await mag_mgr.execute_safe_query(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name = :s",
            params={"s": schema_name},
            limit=1,
        )
        return len(rows) > 0

    # Hardcoded shared schemas (graceful skip if absent)
    for schema_id, desc, logical_app in [
        ("mag",     "MAG admin portal",          "sd-mag"),
        ("oasis",   "Base/reference schema",      "sd-oasis-core"),
        ("company", "Company/vendor service",     "company"),
        ("epds",    "EPDS schema",                "epds"),
    ]:
        if await schema_exists(schema_id):
            _data_sources.append({
                "id":          schema_id,
                "description": desc,
                "logical_app": logical_app,
                "type":        "shared",
            })
        else:
            logger.warning(f"Schema '{schema_id}' not found on server — skipping")

    # OEM tenant schemas from oasis.oem (oem table lives in oasis, not mag)
    oasis_mgr = DatabaseManager(make_url("oasis"))
    try:
        oem_rows = await oasis_mgr.execute_safe_query(
            f"SELECT id, code_name, {_OEM_DISPLAY_COL} FROM oem ORDER BY code_name",
            limit=1000,
        )
        for row in oem_rows:
            code_name    = row.get("code_name", "")
            display_name = row.get(_OEM_DISPLAY_COL, "") or code_name
            tenant_schema = f"oasis_{code_name}"
            if await schema_exists(tenant_schema):
                _data_sources.append({
                    "id":          tenant_schema,
                    "description": f"Tenant: {display_name}",
                    "logical_app": "sd-oasis",
                    "type":        "tenant",
                    "code_name":   code_name,
                    "oem_id":      row.get("id"),
                })
    except Exception as e:
        logger.warning(
            f"Could not query oasis.oem (display_col='{_OEM_DISPLAY_COL}'): {e}. "
            f"Tenant schemas not registered."
        )

    # Cache mag manager for immediate use by tools
    _managers["mag"] = mag_mgr

    logger.info(
        f"Registered {len(_data_sources)} data source(s): "
        f"{[s['id'] for s in _data_sources]}"
    )


async def initialize_symbol_index():
    """Configure the symbol index path. Non-fatal if the file is missing."""
    global _symbol_db_path
    path = os.environ.get("SYMBOL_INDEX_PATH", "/data/symbol_index.db")
    if Path(path).exists():
        _symbol_db_path = path
        logger.info(f"Symbol index loaded: {path}")
    else:
        logger.warning(
            f"Symbol index not found at '{path}' — symbol tools disabled. "
            f"Build with mcp-indexed-tabs/index_repo.sh, then call reload_symbol_index."
        )


async def startup(args):
    """Run all initialization steps."""
    await initialize_data_sources()
    await initialize_symbol_index()


# ---------------------------------------------------------------------------
# DB tools (6)
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_data_sources() -> str:
    """
    List all available data sources (schemas) that can be queried.

    Shows each source's ID, type (shared/tenant), and description.
    The active schema (used by schema_overview, describe_table, query) is marked *.
    Use use_data_source(id) to switch.
    """
    if not _data_sources:
        return "No data sources registered. Check BASE_URL and server startup logs."
    lines = [f"{'ID':<35} {'TYPE':<8} DESCRIPTION"]
    lines.append("-" * 80)
    for ds in _data_sources:
        marker = " *" if ds["id"] == _active_schema_id else ""
        lines.append(f"{ds['id']:<35} {ds['type']:<8} {ds['description']}{marker}")
    lines.append(f"\n* = active schema  (current: {_active_schema_id})")
    return "\n".join(lines)


@mcp.tool()
async def use_data_source(data_source_id: str) -> str:
    """
    Switch the active schema for subsequent schema_overview, describe_table, and query calls.

    Args:
        data_source_id: Schema ID from list_data_sources (e.g. 'mag', 'oasis_tss').

    Returns:
        Confirmation, or error if the ID is not in the registered allowlist.
    """
    global _active_schema_id
    known_ids = {ds["id"] for ds in _data_sources}
    if data_source_id not in known_ids:
        return (
            f"Unknown data source: '{data_source_id}'. "
            f"Known IDs: {', '.join(sorted(known_ids))}"
        )
    _active_schema_id = data_source_id
    return f"Active schema set to: {data_source_id}"


@mcp.tool()
async def schema_overview(filter: str = "", include_row_estimates: bool = False) -> str:
    """
    List tables in the active schema.

    Args:
        filter: Optional substring to filter table names.
        include_row_estimates: Include approximate row counts (MySQL only; may be stale).

    Returns:
        Table list with column counts, and row estimates if requested.
    """
    try:
        mgr = _get_active_manager()
        tables = await mgr.list_tables(filter=filter, include_row_estimates=include_row_estimates)
        if not tables:
            filter_note = f" matching '{filter}'" if filter else ""
            return f"No tables found{filter_note} in schema '{_active_schema_id}'."

        lines = [f"Schema: {_active_schema_id}  ({len(tables)} tables)\n"]
        header = f"{'TABLE':<50} {'COLS':>5}"
        if include_row_estimates:
            header += f"  {'~ROWS':>12}"
        lines.append(header)
        lines.append("-" * (58 if not include_row_estimates else 73))
        for t in tables:
            row = f"{t['table_name']:<50} {t['column_count']:>5}"
            if include_row_estimates and "row_estimate" in t:
                row += f"  {t['row_estimate']:>12,}"
            lines.append(row)
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing tables in '{_active_schema_id}': {e}"


@mcp.tool()
async def describe_table(
    table: str,
    include_indexes: bool = True,
    include_fks: bool = True,
) -> str:
    """
    Show columns, indexes, and foreign keys for a table in the active schema.

    Args:
        table: Table name.
        include_indexes: Include index definitions (MySQL; default True).
        include_fks: Include foreign key constraints (MySQL; default True).

    Returns:
        Structured text block with columns, indexes, and FKs.
    """
    try:
        mgr = _get_active_manager()
        info = await mgr.describe_table(
            table, include_indexes=include_indexes, include_fks=include_fks
        )
        lines = [f"## {table}  (schema: {_active_schema_id})\n"]

        # Columns
        lines.append("### Columns")
        for col in info["columns"]:
            null_note = "NULL" if col["is_nullable"] else "NOT NULL"
            lines.append(f"  {col['column_name']:<35} {col['data_type']:<20} {null_note}")

        # Indexes
        if info.get("indexes"):
            lines.append("\n### Indexes")
            for idx_name, idx_info in info["indexes"].items():
                unique_note = " [UNIQUE]" if idx_info["unique"] else ""
                cols = ", ".join(idx_info["columns"])
                lines.append(f"  {idx_name:<35} ({cols}){unique_note}")

        # Foreign keys
        if info.get("foreign_keys"):
            lines.append("\n### Foreign Keys")
            for fk in info["foreign_keys"]:
                lines.append(
                    f"  {fk['column']:<30} → "
                    f"{fk['references_table']}.{fk['references_column']}"
                    f"  [{fk['constraint_name']}]"
                )

        return "\n".join(lines)
    except Exception as e:
        return f"Error describing table '{table}' in '{_active_schema_id}': {e}"


@mcp.tool()
async def query(
    sql: str,
    params: Optional[Dict[str, Any]] = None,
    max_rows: int = 200,
    timeout_ms: int = 3000,
    data_source_id: Optional[str] = None,
) -> str:
    """
    Execute a read-only SQL query against the active schema (or a named schema).

    Only SELECT queries (and WITH…SELECT) are permitted. Mutating keywords raise an error.

    Args:
        sql: SELECT query. Use :param_name for bind parameters.
        params: Named bind parameters dict (e.g. {"id": 42, "name": "foo%"}).
        max_rows: Maximum rows to return (hard cap 1000; default 200).
        timeout_ms: Query timeout in milliseconds (default 3000).
        data_source_id: Override active schema for this call only (does not change active schema).

    Returns:
        Column headers and rows formatted as aligned text, plus truncation note if capped.
    """
    max_rows = min(max_rows, 1000)

    known_ids = {ds["id"] for ds in _data_sources}
    if data_source_id is not None:
        if data_source_id not in known_ids:
            return (
                f"Unknown data source: '{data_source_id}'. "
                f"Known: {', '.join(sorted(known_ids))}"
            )
        if data_source_id not in _managers:
            _managers[data_source_id] = DatabaseManager(make_url(data_source_id))
        mgr = _managers[data_source_id]
    else:
        mgr = _get_active_manager()

    try:
        rows = await mgr.execute_safe_query(
            sql, params=params, limit=max_rows, timeout_ms=timeout_ms
        )
    except Exception as e:
        return f"Query error: {e}"

    if not rows:
        return "Query returned no rows."

    headers = list(rows[0].keys())
    col_widths = [
        max(len(h), max(len(str(r.get(h, ""))) for r in rows))
        for h in headers
    ]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line    = "  ".join("-" * w for w in col_widths)
    data_lines  = [
        "  ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, col_widths))
        for r in rows
    ]

    output = "\n".join([header_line, sep_line] + data_lines)
    if len(rows) == max_rows:
        output += f"\n\n(results capped at {max_rows} rows)"
    return output


@mcp.tool()
async def resolve_tenant(q: str) -> str:
    """
    Find OEM tenant(s) by code name or display name substring.

    Only returns tenants that have a registered oasis_<code_name> schema.
    Use the returned data_source_id with use_data_source() to switch to that tenant.

    Args:
        q: Search string (code name or display name fragment; auto-wrapped in % wildcards).

    Returns:
        Matching tenants: tenant_key, data_source_id, display_name.
    """
    if "oasis" not in _managers:
        _managers["oasis"] = DatabaseManager(make_url("oasis"))
    oasis_mgr = _managers["oasis"]

    pattern = q if ("%" in q or "_" in q) else f"%{q}%"
    try:
        rows = await oasis_mgr.execute_safe_query(
            f"SELECT id, code_name, {_OEM_DISPLAY_COL} FROM oem "
            f"WHERE code_name LIKE :q OR {_OEM_DISPLAY_COL} LIKE :q "
            f"LIMIT 10",
            params={"q": pattern},
            limit=10,
        )
    except Exception as e:
        return f"Error querying oasis.oem: {e}"

    tenant_ids = {ds["id"] for ds in _data_sources if ds.get("type") == "tenant"}
    results = []
    for row in rows:
        code_name = row.get("code_name", "")
        ds_id     = f"oasis_{code_name}"
        if ds_id in tenant_ids:
            results.append({
                "tenant_key":    code_name,
                "data_source_id": ds_id,
                "display_name":  row.get(_OEM_DISPLAY_COL, "") or code_name,
            })

    if not results:
        return f"No tenants found matching '{q}'."

    lines = [f"{'TENANT KEY':<25} {'DATA SOURCE ID':<35} DISPLAY NAME"]
    lines.append("-" * 80)
    for r in results:
        lines.append(
            f"{r['tenant_key']:<25} {r['data_source_id']:<35} {r['display_name']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Symbol index tools — ported verbatim from mcp-indexed-tabs/symbol_index_mcp.py
# Each tool returns early with a friendly message if _symbol_db_path is None.
# ---------------------------------------------------------------------------

@mcp.tool()
def find_symbol(
    name: str,
    kind: str = "",
    repo: str = "",
    scope: str = "",
    annotation: str = "",
    offset: int = 0,
) -> str:
    """
    Search for symbols (classes, methods, fields, etc.) by name and/or owning class.
    Supports SQL LIKE wildcards: % matches any sequence, _ matches one char.
    If no wildcard is present, wraps the name in % for a substring match.

    Args:
        name: Symbol name or partial name to search for.
        kind: Optional filter — class, method, field, interface, enum, etc.
        repo: Optional filter — sd-mag or sd-oasis-core.
        scope: Optional owning class filter. Pass a class name (e.g. "BatchUpdateService")
               to list all symbols belonging to that class.
        annotation: Optional annotation filter (e.g. "Secured", "Transactional").
                    Auto-wrapped in % if no wildcard present.
        offset: Skip the first N results (for pagination). Default 0.

    Returns:
        Matching symbols with file path, line number, kind, and repo.
    """
    if _symbol_db_path is None:
        return "Symbol index not available."
    if not name:
        return "Error: name is required."

    pattern = name if ("%" in name or "_" in name) else f"%{name}%"

    clauses = ["s.name LIKE :name", "s.is_qualified = 0"]
    params: dict = {"name": pattern, "offset": offset}

    if kind:
        clauses.append("s.kind = :kind")
        params["kind"] = kind
    if repo:
        clauses.append("f.repo = :repo")
        params["repo"] = repo
    if scope:
        clauses.append("s.scope = :scope")
        params["scope"] = scope
    if annotation:
        ann_pattern = annotation if ("%" in annotation or "_" in annotation) else f"%{annotation}%"
        clauses.append(
            "f.id IN (SELECT file_id FROM annotations WHERE annotation_name LIKE :ann)"
        )
        params["ann"] = ann_pattern

    sql = f"""
        SELECT s.name, s.kind, s.access, f.repo, f.artifact_type, f.path, s.line, s.scope
        FROM symbols s
        JOIN files f ON s.file_id = f.id
        WHERE {' AND '.join(clauses)}
        ORDER BY f.repo, s.kind, s.name
        LIMIT 100 OFFSET :offset
    """

    with _sym_conn() as con:
        rows = con.execute(sql, params).fetchall()

    if not rows:
        return f"No symbols found matching '{name}'."

    lines = [f"{'NAME':<40} {'KIND':<12} {'ACCESS':<10} {'REPO':<16} {'ARTIFACT':<14} FILE:LINE"]
    lines.append("-" * 120)
    for r in rows:
        scope_note = f"  (in {r['scope']})" if r["scope"] and not scope else ""
        lines.append(
            f"{r['name']:<40} {(r['kind'] or ''):<12} {(r['access'] or ''):<10} {r['repo']:<16} "
            f"{(r['artifact_type'] or ''):<14} {r['path']}:{r['line']}{scope_note}"
        )

    truncated = ""
    if len(rows) == 100:
        next_offset = offset + 100
        truncated = f"\n(showing {offset + 1}–{offset + 100} — use offset={next_offset} for next page)"

    return "\n".join(lines) + truncated


@mcp.tool()
def describe_class(class_name: str, usages: bool = False) -> str:
    """
    Return full details for a Grails class: location, artifact type, GORM relations,
    class-level annotations, injected dependencies, methods, and constraint field count.
    Also shows extends/implements hierarchy, per-method annotations, and job trigger expressions.

    Args:
        class_name: Exact class name (e.g. BatchUpdateService, DetailPart).
        usages: When True, also show reverse lookups — reverse DI, reverse GORM relations,
                and subclasses/implementors. Useful for impact analysis.

    Returns:
        Structured summary of the class across all index tables.
    """
    if _symbol_db_path is None:
        return "Symbol index not available."
    if not class_name:
        return "Error: class_name is required."

    out = []

    with _sym_conn() as con:
        # Basic symbol + file info
        row = con.execute("""
            SELECT s.name, s.kind, s.line, s.access,
                   f.repo, f.path, f.artifact_type, f.package
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE s.name = :n
              AND s.kind IN ('class', 'interface', 'enum', 'trait')
              AND s.is_qualified = 0
            LIMIT 1
        """, {"n": class_name}).fetchone()

        if not row:
            return f"Class '{class_name}' not found. Try find_symbol to locate it."

        out.append(f"## {row['name']}  [{row['kind']}]")
        out.append(f"Repo:      {row['repo']}")
        out.append(f"Artifact:  {row['artifact_type'] or '(unknown)'}")
        out.append(f"Package:   {row['package'] or '(unknown)'}")
        out.append(f"File:      {row['path']}:{row['line']}")

        max_line_row = con.execute(
            "SELECT MAX(line) FROM symbols WHERE file_id = "
            "(SELECT id FROM files WHERE path=:p AND repo=:r)",
            {"p": row["path"], "r": row["repo"]}
        ).fetchone()
        if max_line_row and max_line_row[0]:
            approx_lines = max_line_row[0]
            size_note = "  ⚠ large file — read by targeted line range" if approx_lines >= 500 else ""
            out.append(f"Lines:     ~{approx_lines}{size_note}")

        file_id_row = con.execute(
            "SELECT id FROM files WHERE path = :p AND repo = :r",
            {"p": row["path"], "r": row["repo"]}
        ).fetchone()
        file_id = file_id_row["id"] if file_id_row else None

        # Class hierarchy
        hierarchy = con.execute("""
            SELECT relation, target FROM class_hierarchy
            WHERE class_name = :c ORDER BY relation DESC, target
        """, {"c": class_name}).fetchall()

        extends_targets    = [h["target"] for h in hierarchy if h["relation"] == "extends"]
        implements_targets = [h["target"] for h in hierarchy if h["relation"] == "implements"]

        out.append(f"Extends:   {extends_targets[0] if extends_targets else '(none)'}")
        if implements_targets:
            out.append(f"Implements:{', '.join(implements_targets)}")

        # Job triggers
        if row["artifact_type"] == "job":
            triggers = con.execute("""
                SELECT trigger_type, expression FROM job_triggers
                WHERE class_name = :c ORDER BY trigger_type, expression
            """, {"c": class_name}).fetchall()
            if triggers:
                first = True
                for t in triggers:
                    expr_str = f'{t["trigger_type"]}("{t["expression"]}")'
                    if first:
                        out.append(f"Triggers:  {expr_str}")
                        first = False
                    else:
                        out.append(f"           {expr_str}")

        # Annotations (class-level)
        if file_id:
            anns = con.execute("""
                SELECT annotation_name, attributes FROM annotations
                WHERE file_id = :fid ORDER BY annotation_name
            """, {"fid": file_id}).fetchall()
            if anns:
                out.append("\n### Annotations")
                for a in anns:
                    attrs = f"  {a['attributes']}" if a["attributes"] else ""
                    out.append(f"  @{a['annotation_name']}{attrs}")

        # GORM relations
        rels = con.execute("""
            SELECT relation_type, field_name, target_class, extra FROM gorm_relations
            WHERE owner_class = :c ORDER BY relation_type, field_name
        """, {"c": class_name}).fetchall()
        if rels:
            out.append("\n### GORM Relations")
            for r in rels:
                extra = f"  [{r['extra']}]" if r["extra"] else ""
                out.append(
                    f"  {r['relation_type']:<14} {(r['field_name'] or ''):<30} "
                    f"→ {r['target_class'] or ''}{extra}"
                )

        # GORM constraint field count
        constraint_count = con.execute(
            "SELECT COUNT(*) as cnt FROM gorm_constraints WHERE owner_class = :c",
            {"c": class_name}
        ).fetchone()["cnt"]
        if constraint_count:
            out.append(f"\n### Constraints\n  {constraint_count} constrained field(s) defined")

        # Injected dependencies
        deps = con.execute("""
            SELECT dep_name, dep_type, is_typed FROM injected_deps
            WHERE source_class = :c ORDER BY dep_name
        """, {"c": class_name}).fetchall()
        if deps:
            out.append("\n### Injected Dependencies")
            for d in deps:
                typed_label = (
                    d["dep_type"] if d["is_typed"]
                    else f"def  (resolved: {d['dep_type'] or '?'})"
                )
                out.append(f"  {d['dep_name']:<35} {typed_label}")

        # Methods
        if file_id:
            methods = con.execute("""
                SELECT s.name, s.signature, s.access, s.line
                FROM symbols s
                WHERE s.file_id = :fid AND s.scope = :cls
                  AND s.kind = 'method' AND s.is_qualified = 0
                ORDER BY s.name
            """, {"fid": file_id, "cls": class_name}).fetchall()
            if methods:
                m_ann_rows = con.execute("""
                    SELECT method_name, annotation_name, attributes
                    FROM method_annotations
                    WHERE file_id = :fid AND class_name = :cls
                    ORDER BY method_name, annotation_name
                """, {"fid": file_id, "cls": class_name}).fetchall()
                m_ann_map: dict = {}
                for r in m_ann_rows:
                    m_ann_map.setdefault(r["method_name"], []).append(
                        (r["annotation_name"], r["attributes"])
                    )

                out.append("\n### Methods")
                for m in methods:
                    sig        = m["signature"] or "()"
                    access     = m["access"] or ""
                    access_note = f"  [{access}]" if access and access != "default" else ""
                    anns       = m_ann_map.get(m["name"], [])
                    ann_str    = (
                        "  " + " ".join(_format_ann_attr(a, v) for a, v in anns)
                    ) if anns else ""
                    out.append(f"  {m['name']}{sig}  :L{m['line']}{access_note}{ann_str}")

        # Reverse lookups (usages=True)
        if usages:
            reverse_di = con.execute("""
                SELECT d.source_class, d.dep_name, f.artifact_type
                FROM injected_deps d JOIN files f ON d.file_id = f.id
                WHERE d.dep_type = :c ORDER BY d.source_class
            """, {"c": class_name}).fetchall()
            if reverse_di:
                out.append("\n### Injected Into (reverse DI)")
                for d in reverse_di:
                    out.append(f"  {d['source_class']:<40} as {d['dep_name']}  [{d['artifact_type']}]")
            else:
                out.append("\n### Injected Into (reverse DI)\n  (none found)")

            reverse_gorm = con.execute("""
                SELECT g.owner_class, g.relation_type, g.field_name, f.artifact_type
                FROM gorm_relations g JOIN files f ON g.file_id = f.id
                WHERE g.target_class = :c ORDER BY g.owner_class
            """, {"c": class_name}).fetchall()
            if reverse_gorm:
                out.append("\n### GORM Relations To This Class")
                for g in reverse_gorm:
                    field = f" via {g['field_name']}" if g["field_name"] else ""
                    out.append(
                        f"  {g['owner_class']:<40} {g['relation_type']}{field}"
                        f"  [{g['artifact_type']}]"
                    )
            else:
                out.append("\n### GORM Relations To This Class\n  (none found)")

            subclasses = con.execute("""
                SELECT ch.class_name, ch.relation, f.artifact_type
                FROM class_hierarchy ch JOIN files f ON ch.file_id = f.id
                WHERE ch.target = :c ORDER BY ch.relation, ch.class_name
            """, {"c": class_name}).fetchall()
            if subclasses:
                out.append("\n### Subclasses / Implementors")
                for sc in subclasses:
                    out.append(
                        f"  {sc['class_name']:<35} {sc['relation']:<12} [{sc['artifact_type']}]"
                    )
            else:
                out.append("\n### Subclasses / Implementors\n  (none found)")

    return "\n".join(out)


@mcp.tool()
def find_route(pattern: str) -> str:
    """
    Find URL mappings by controller name or URL pattern.
    Supports SQL LIKE wildcards; auto-wraps in % if none present.

    Args:
        pattern: Controller name or URL fragment (e.g. "BatchUpdate", "/api/").

    Returns:
        Matching URL mappings with HTTP method, URL pattern, controller, and action.
    """
    if _symbol_db_path is None:
        return "Symbol index not available."
    if not pattern:
        return "Error: pattern is required."

    like = pattern if ("%" in pattern or "_" in pattern) else f"%{pattern}%"
    sql = """
        SELECT url_pattern, http_method, controller, action, view, f.repo
        FROM url_mappings u
        JOIN files f ON u.file_id = f.id
        WHERE u.controller LIKE :p OR u.url_pattern LIKE :p
        ORDER BY f.repo, u.controller, u.action
        LIMIT 50
    """

    with _sym_conn() as con:
        rows = con.execute(sql, {"p": like}).fetchall()

    if not rows:
        return f"No URL mappings found for '{pattern}'."

    lines = [f"{'HTTP':<8} {'URL PATTERN':<40} {'CONTROLLER':<30} {'ACTION':<25} REPO"]
    lines.append("-" * 120)
    for r in rows:
        method   = r["http_method"] or "*"
        view_str = f"  → view:{r['view']}" if r["view"] else ""
        lines.append(
            f"{method:<8} {r['url_pattern']:<40} {(r['controller'] or ''):<30} "
            f"{(r['action'] or ''):<25} {r['repo']}{view_str}"
        )

    return "\n".join(lines)


@mcp.tool()
async def reload_symbol_index() -> str:
    """
    Reload the symbol index from SYMBOL_INDEX_PATH.

    Use this after rebuilding data/symbol_index.db with mcp-indexed-tabs/index_repo.sh.
    No server restart required.

    Returns:
        Status, path, file size in MB, and last-modified timestamp.
    """
    global _symbol_db_path
    import datetime

    path_str = os.environ.get("SYMBOL_INDEX_PATH", "/data/symbol_index.db")
    p = Path(path_str)
    if not p.exists():
        return f"File not found: {path_str}"

    stat      = p.stat()
    _symbol_db_path = path_str
    mtime     = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
    size_mb   = stat.st_size / (1024 * 1024)

    return (
        f"status:    loaded\n"
        f"path:      {path_str}\n"
        f"size_mb:   {size_mb:.2f}\n"
        f"modified:  {mtime}"
    )


# ---------------------------------------------------------------------------
# MCP Resources (updated to use active schema)
# ---------------------------------------------------------------------------

@mcp.resource("database://tables")
async def get_database_tables() -> str:
    """Resource: table listing for the active schema."""
    mgr    = _get_active_manager()
    tables = await mgr.list_tables()
    lines  = [f"- {t['table_name']} ({t['column_count']} columns)" for t in tables]
    return f"Schema: {_active_schema_id}\n\n" + "\n".join(lines)


@mcp.resource("database://schema")
async def get_database_schema() -> str:
    """Resource: full schema for the active schema as JSON."""
    mgr    = _get_active_manager()
    tables = await mgr.list_tables()
    schema_info = {}
    for t in tables:
        schema_info[t["table_name"]] = await mgr.describe_table(t["table_name"])
    return json.dumps(schema_info, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA MCP Server — DB + Symbol Index")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport: stdio (default) or http (streamable-HTTP on 0.0.0.0:8000)",
    )
    args = parser.parse_args()

    async def run():
        await startup(args)
        if args.transport == "http":
            await mcp.run_streamable_http_async()
        else:
            await mcp.run_stdio_async()

    asyncio.run(run())

# QA MCP Server — DB + Symbol Index

An MCP server for the Oasis QA team. Two capability sets in one container:

1. **DB tools (6)** — multi-schema MySQL navigation against QA1
2. **Symbol index tools (4)** — Grails codebase exploration (ported from `mcp-indexed-tabs`)

Claude Code connects via streamable-HTTP. No credentials are ever sent to the agent — only schema IDs.

---

## Prerequisites

- Docker Desktop
- QA1 MySQL credentials (ask team lead)
- `data/symbol_index.db` — built with `mcp-indexed-tabs/index_repo.sh` (optional; symbol tools disabled if absent)

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd mcp-db-server

# 2. Fill in credentials
cp .env.example .env
# Edit .env: set BASE_URL=mysql+aiomysql://USER:PASS@host.docker.internal:3307/

# 3. Build symbol index (optional but recommended)
#    Run from the workspace root that contains sd-mag/ and sd-oasis-core/
mkdir -p data
# Follow instructions in ../mcp-indexed-tabs/README.md to build data/symbol_index.db

# 4. Start
docker-compose up --build

# 5. Verify
docker ps   # qa-mcp-server should be "Up"
```

---

## Claude Code configuration

Add to `~/.claude.json` (or Claude Desktop config):

```json
{
  "mcpServers": {
    "qa-db":      { "url": "http://localhost:8765/db/mcp" },
    "qa-symbols": { "url": "http://localhost:8765/symbols/mcp" }
  }
}
```

If you changed `MCP_PORT` in `.env`, update the port in both URLs accordingly.

---

## DB tools

| Tool | Description |
|------|-------------|
| `list_data_sources()` | List all registered schemas (mag, oasis, oasis_*, company, epds) |
| `use_data_source(id)` | Switch the active schema for subsequent calls |
| `schema_overview([filter], [include_row_estimates])` | List tables in the active schema |
| `describe_table(table, [include_indexes], [include_fks])` | Columns + indexes + foreign keys |
| `query(sql, [params], [max_rows], [timeout_ms], [data_source_id])` | Execute a read-only SELECT |
| `resolve_tenant(q)` | Find tenant by code name or display name; returns data_source_id |

**Schema naming convention:**

| ID | Description |
|----|-------------|
| `mag` | MAG admin portal (sd-mag) |
| `oasis` | Base/reference schema (sd-oasis-core) |
| `oasis_{code_name}` | OEM tenant (e.g. `oasis_tss`, `oasis_acme`) |
| `company` | Company/vendor microservice |
| `epds` | EPDS schema |

**Example session:**

```
list_data_sources()
use_data_source("oasis_tss")
schema_overview(filter="part")
describe_table("part")
query("SELECT id, part_number FROM part LIMIT 5")
resolve_tenant("tss")
```

---

## Symbol index tools

| Tool | Description |
|------|-------------|
| `find_symbol(name, [kind], [repo], [scope], [annotation], [offset])` | Search symbols by name/annotation |
| `describe_class(class_name, [usages])` | Full class details + optional reverse lookups |
| `find_route(pattern)` | URL mapping lookup by controller or URL fragment |
| `reload_symbol_index()` | Hot-reload after rebuilding `data/symbol_index.db` |

**Example session:**

```
find_symbol(name="BatchUpdateService", kind="class")
describe_class("BatchUpdateService", usages=True)
find_route(pattern="batchUpdate")
```

---

## Re-indexing (symbol index)

```bash
# From workspace root (contains sd-mag/, sd-oasis-core/)
cd ../mcp-indexed-tabs
bash index_repo.sh ../sd-mag ../sd-oasis-core ../mcp-db-server/data/symbol_index.db

# Then hot-reload (no restart needed)
# In Claude Code: reload_symbol_index()
```

---

## Optional debug REST layer

The original FastAPI HTTP endpoints are available in a debug profile:

```bash
docker-compose --profile debug up
# Swagger UI: http://localhost:8001/docs
```

Note: `describe_table` response format changed in this fork — the debug server's
`/mcp/describe/{table_name}` endpoint may not format correctly.

---

## Troubleshooting

**Linux: cannot reach host MySQL**
Add to `mcp-server` in `docker-compose.yml`:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

**MySQL SSL errors**
Set `MYSQL_SSL=true` in `.env` if your QA1 instance requires SSL.

**`_OEM_DISPLAY_COL` query fails at startup**
The display-name column in `oasis.oem` defaults to `name`. If startup logs warn about
`Could not query oasis.oem`, check the actual column name with:
```
query("SELECT * FROM oem LIMIT 1", data_source_id="oasis")
```
Then update `_OEM_DISPLAY_COL` in `mcp_server.py` and rebuild.

**Symbol tools return "Symbol index not available"**
Build `data/symbol_index.db` (see Re-indexing above), then call `reload_symbol_index()`.

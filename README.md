# codex-research-mcp

A small local MCP adapter that lets Claude Code delegate bounded research work to
Codex plan credits without giving the worker ownership of the final report.

## Why it exists

Claude Code only schedules MCP tools concurrently when they advertise
`readOnlyHint: true`. The official generic Codex MCP tools can write to arbitrary
workspaces, so they cannot safely make that promise. This adapter exposes a
narrower contract:

- `research` starts an independent Codex research thread;
- `research_reply` continues a thread created during the current MCP server
  lifetime;
- real source trees remain read-only;
- every worker receives a private writable scratch directory under `/tmp`;
- network reads are enabled, while approvals and durable external writes are not;
- both tools advertise the read-only annotation so independent Claude Code tool
  calls can run concurrently.

The adapter forwards execution to one long-lived official `codex mcp-server`.
Codex still owns model selection, authentication, sessions, and plan-credit use.

## Run from this checkout

```sh
./bin/codex-research-mcp
```

Example Claude Code MCP entry:

```json
{
  "research-worker": {
    "type": "stdio",
    "command": "/Users/zhuhuibin/lab/codex-research-mcp/bin/codex-research-mcp",
    "args": [],
    "env": {
      "CODEX_RESEARCH_MAX_CONCURRENCY": "4"
    }
  }
}
```

Useful optional environment variables:

- `CODEX_RESEARCH_CODEX_BIN`: Codex executable, default `codex`;
- `CODEX_RESEARCH_MAX_CONCURRENCY`: simultaneous Codex calls, default `4`;
- `CODEX_RESEARCH_TIMEOUT_SECONDS`: per-call timeout, default `3600`;
- `CODEX_RESEARCH_STATE_DIR`: scratch and authorized-thread state, default
  `<system temporary directory>/codex-research-mcp`;
- `CODEX_RESEARCH_SOURCE_ROOT`: default readable source root when a tool call does
  not provide `source_cwd`.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

"""A narrow MCP adapter that exposes Codex as a parallel research worker."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable


PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "codex-research-mcp"
SERVER_VERSION = "0.1.0"
DEFAULT_STATE_DIR = Path(tempfile.gettempdir()) / SERVER_NAME
THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{10,128}$")

RESEARCH_TOOL = {
    "name": "research",
    "title": "Codex Research",
    "description": (
        "Delegate one bounded, read-only research assignment to Codex. The real "
        "source tree stays read-only; downloads, scripts, and intermediate files "
        "are confined to a private temporary scratch directory. Independent calls "
        "may run concurrently."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "description": "The complete bounded research assignment.",
            },
            "source_cwd": {
                "type": "string",
                "description": (
                    "Absolute source root for local material. It is readable but is "
                    "not the worker's writable workspace. Defaults to the MCP "
                    "server's startup directory."
                ),
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
    "outputSchema": {
        "type": "object",
        "properties": {
            "threadId": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["threadId", "content"],
    },
    "annotations": {
        "title": "Read-only Codex research",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
}

RESEARCH_REPLY_TOOL = {
    "name": "research_reply",
    "title": "Continue Codex Research",
    "description": (
        "Continue a research thread previously created during the current MCP server "
        "lifetime. The original scratch isolation and source boundary remain in force."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "threadId": {
                "type": "string",
                "description": "A thread id returned by the research tool.",
            },
            "prompt": {
                "type": "string",
                "minLength": 1,
                "description": "The follow-up research instruction.",
            },
        },
        "required": ["threadId", "prompt"],
        "additionalProperties": False,
    },
    "outputSchema": RESEARCH_TOOL["outputSchema"],
    "annotations": {
        "title": "Continue read-only Codex research",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
}

TOOLS = [RESEARCH_TOOL, RESEARCH_REPLY_TOOL]


class UpstreamError(RuntimeError):
    """Raised when the official Codex MCP server fails a request."""


class CodexMcpClient:
    """Minimal asynchronous JSON-RPC client for `codex mcp-server`."""

    def __init__(self, command: str, timeout_seconds: float) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.process: asyncio.subprocess.Process | None = None
        self.pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self.next_id = 1
        self.write_lock = asyncio.Lock()
        self.stdout_task: asyncio.Task[None] | None = None
        self.stderr_task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self) -> None:
        if self.is_running:
            return
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            "mcp-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.stdout_task = asyncio.create_task(self._read_stdout())
        self.stderr_task = asyncio.create_task(self._relay_stderr())
        try:
            response = await self.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
            if "error" in response:
                raise UpstreamError(str(response["error"]))
            await self.notify("notifications/initialized", {})
        except BaseException:
            await self.close()
            raise

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while line := await self.process.stdout.readline():
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                message_id = message.get("id")
                if message_id not in self.pending:
                    continue
                future = self.pending.pop(message_id)
                if not future.done():
                    future.set_result(message)
        finally:
            error = UpstreamError("Codex MCP server closed its output stream")
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)
            self.pending.clear()

    async def _relay_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while chunk := await self.process.stderr.read(65536):
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.is_running or self.process is None or self.process.stdin is None:
            raise UpstreamError("Codex MCP server is not running")
        message_id = self.next_id
        self.next_id += 1
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self.pending[message_id] = future
        payload = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": method,
            "params": params,
        }
        try:
            async with self.write_lock:
                self.process.stdin.write(
                    (json.dumps(payload, separators=(",", ":")) + "\n").encode()
                )
                await self.process.stdin.drain()
            return await asyncio.wait_for(future, timeout=self.timeout_seconds)
        except BaseException as error:
            self.pending.pop(message_id, None)
            if not future.done():
                future.cancel()
            if self.is_running:
                try:
                    await asyncio.shield(
                        asyncio.wait_for(
                            self.notify(
                                "notifications/cancelled",
                                {
                                    "requestId": message_id,
                                    "reason": type(error).__name__,
                                },
                            ),
                            timeout=1,
                        )
                    )
                except BaseException:
                    pass
            raise

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        if not self.is_running or self.process is None or self.process.stdin is None:
            raise UpstreamError("Codex MCP server is not running")
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        async with self.write_lock:
            self.process.stdin.write(
                (json.dumps(payload, separators=(",", ":")) + "\n").encode()
            )
            await self.process.stdin.drain()

    async def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None and not self.process.stdin.is_closing():
            self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=3)
        except asyncio.TimeoutError:
            self.process.terminate()
            await self.process.wait()
        for task in (self.stdout_task, self.stderr_task):
            if task is not None:
                await task


UpstreamFactory = Callable[[], CodexMcpClient]


class ResearchService:
    """Translate research-only tools into constrained official Codex calls."""

    def __init__(
        self,
        state_dir: Path,
        startup_cwd: Path,
        max_concurrency: int,
        upstream_factory: UpstreamFactory,
    ) -> None:
        self.state_dir = state_dir.resolve()
        self.startup_cwd = startup_cwd.resolve()
        self.scratch_root = self.state_dir / "scratch"
        self.thread_root = self.state_dir / "threads"
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.thread_root.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.upstream_factory = upstream_factory
        self.upstream: CodexMcpClient | None = None
        self.upstream_lock = asyncio.Lock()
        self.authorized_threads: set[str] = set()

    async def _get_upstream(self) -> CodexMcpClient:
        async with self.upstream_lock:
            if self.upstream is not None and self.upstream.is_running:
                return self.upstream
            if self.upstream is not None:
                await self.upstream.close()
            self.upstream = self.upstream_factory()
            await self.upstream.start()
            return self.upstream

    def _source_root(self, value: Any) -> Path:
        if value is None:
            source_root = self.startup_cwd
        elif not isinstance(value, str) or not value.strip():
            raise ValueError("source_cwd must be a non-empty absolute path")
        else:
            source_root = Path(value).expanduser()
            if not source_root.is_absolute():
                raise ValueError("source_cwd must be an absolute path")
            source_root = source_root.resolve()
        if not source_root.is_dir():
            raise ValueError(f"source_cwd is not a directory: {source_root}")
        return source_root

    def _thread_path(self, thread_id: str) -> Path:
        if not THREAD_ID_PATTERN.fullmatch(thread_id):
            raise ValueError("invalid research thread id")
        return self.thread_root / f"{thread_id}.json"

    def _remember_thread(
        self, thread_id: str, source_root: Path, scratch_dir: Path
    ) -> None:
        thread_path = self._thread_path(thread_id)
        payload = {
            "threadId": thread_id,
            "sourceRoot": str(source_root),
            "scratchDir": str(scratch_dir),
        }
        temporary_path = thread_path.with_suffix(f".{os.getpid()}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, thread_path)
        self.authorized_threads.add(thread_id)

    @staticmethod
    def _developer_instructions(source_root: Path, scratch_dir: Path) -> str:
        return f"""You are a bounded research worker supporting a lead researcher.

Safety and ownership boundaries:
- Treat every real user, project, research-library, and repository path as read-only.
- The source root is: {source_root}
- Your only writable workspace is the private scratch directory: {scratch_dir}
- Put downloads, scripts, OCR output, caches, extracts, and all intermediate artifacts in that scratch directory.
- Do not edit durable user deliverables, source material, repositories, configuration, credentials, or application state.
- Do not send messages, publish content, install system packages, or make external state changes.
- Network reads, Hubble retrieval, web research, and local computation are allowed.
- Resolve relative source references against the source root above.

Research role:
- Complete only the bounded assignment in the user prompt.
- Return a compact evidence packet with exact URLs or absolute local paths, source dates, confidence, conflicts, and gaps.
- Prefer original sources. You are not yourself a source.
- Do not write the lead researcher's thesis, recommendation, outline, or final report prose.
"""

    @staticmethod
    def _validate_prompt(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("prompt must be a non-empty string")
        return value

    async def research(self, arguments: dict[str, Any]) -> dict[str, Any]:
        prompt = self._validate_prompt(arguments.get("prompt"))
        source_root = self._source_root(arguments.get("source_cwd"))
        scratch_dir = Path(
            tempfile.mkdtemp(prefix="task-", dir=self.scratch_root)
        ).resolve()
        async with self.semaphore:
            upstream = await self._get_upstream()
            response = await upstream.request(
                "tools/call",
                {
                    "name": "codex",
                    "arguments": {
                        "prompt": prompt,
                        "approval-policy": "never",
                        "sandbox": "workspace-write",
                        "cwd": str(scratch_dir),
                        "developer-instructions": self._developer_instructions(
                            source_root, scratch_dir
                        ),
                        "config": {
                            "sandbox_workspace_write.network_access": True,
                        },
                    },
                },
            )
        result = self._unwrap_upstream(response)
        thread_id = self._thread_id_from_result(result)
        self._remember_thread(thread_id, source_root, scratch_dir)
        return result

    async def research_reply(self, arguments: dict[str, Any]) -> dict[str, Any]:
        prompt = self._validate_prompt(arguments.get("prompt"))
        thread_id = arguments.get("threadId")
        if not isinstance(thread_id, str):
            raise ValueError("threadId must be a string")
        if thread_id not in self.authorized_threads:
            raise ValueError(
                "unknown research thread id in this MCP session; start it with the "
                "research tool first"
            )
        async with self.semaphore:
            upstream = await self._get_upstream()
            response = await upstream.request(
                "tools/call",
                {
                    "name": "codex-reply",
                    "arguments": {"threadId": thread_id, "prompt": prompt},
                },
            )
        result = self._unwrap_upstream(response)
        returned_thread_id = self._thread_id_from_result(result)
        if returned_thread_id != thread_id:
            raise UpstreamError("Codex returned a different research thread id")
        return result

    @staticmethod
    def _unwrap_upstream(response: dict[str, Any]) -> dict[str, Any]:
        if "error" in response:
            raise UpstreamError(json.dumps(response["error"], ensure_ascii=False))
        result = response.get("result")
        if not isinstance(result, dict):
            raise UpstreamError("Codex MCP returned an invalid tool result")
        return result

    @staticmethod
    def _thread_id_from_result(result: dict[str, Any]) -> str:
        structured = result.get("structuredContent")
        thread_id = structured.get("threadId") if isinstance(structured, dict) else None
        if not isinstance(thread_id, str) or not THREAD_ID_PATTERN.fullmatch(thread_id):
            raise UpstreamError("Codex MCP result did not contain a valid threadId")
        return thread_id

    async def close(self) -> None:
        if self.upstream is not None:
            await self.upstream.close()


class McpServer:
    """Concurrent stdio MCP server."""

    def __init__(self, service: ResearchService) -> None:
        self.service = service
        self.write_lock = asyncio.Lock()
        self.tasks: dict[Any, asyncio.Task[None]] = {}

    async def send(self, message: dict[str, Any]) -> None:
        encoded = json.dumps(
            message, ensure_ascii=False, separators=(",", ":")
        )
        async with self.write_lock:
            sys.stdout.write(encoded + "\n")
            sys.stdout.flush()

    async def _progress_heartbeat(self, token: Any) -> None:
        progress = 0
        await self.send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": token,
                    "progress": progress,
                    "message": "Codex research worker started",
                },
            }
        )
        while True:
            await asyncio.sleep(15)
            progress += 1
            await self.send(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": token,
                        "progress": progress,
                        "message": f"Codex research worker is still running ({progress * 15}s)",
                    },
                }
            )

    async def _handle_tool_call(self, message: dict[str, Any]) -> None:
        message_id = message["id"]
        params = message.get("params") or {}
        metadata = params.get("_meta") or {}
        progress_token = metadata.get("progressToken")
        heartbeat = (
            asyncio.create_task(self._progress_heartbeat(progress_token))
            if progress_token is not None
            else None
        )
        try:
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            if name == "research":
                result = await self.service.research(arguments)
            elif name == "research_reply":
                result = await self.service.research_reply(arguments)
            else:
                raise ValueError(f"unknown tool: {name}")
            await self.send(
                {"jsonrpc": "2.0", "id": message_id, "result": result}
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.send(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": [{"type": "text", "text": str(error)}],
                        "isError": True,
                    },
                }
            )
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
            self.tasks.pop(message_id, None)

    async def handle(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        message_id = message.get("id")
        if method == "initialize":
            await self.send(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": SERVER_NAME,
                            "version": SERVER_VERSION,
                        },
                    },
                }
            )
        elif method == "ping":
            await self.send({"jsonrpc": "2.0", "id": message_id, "result": {}})
        elif method == "tools/list":
            await self.send(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {"tools": TOOLS},
                }
            )
        elif method == "tools/call":
            task = asyncio.create_task(self._handle_tool_call(message))
            self.tasks[message_id] = task
        elif method == "notifications/cancelled":
            request_id = (message.get("params") or {}).get("requestId")
            task = self.tasks.get(request_id)
            if task is not None:
                task.cancel()
        elif method in {"notifications/initialized", "notifications/roots/list_changed"}:
            return
        elif message_id is not None:
            await self.send(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            )

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict):
                    await self.handle(message)
        finally:
            if self.tasks:
                tasks = list(self.tasks.values())
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await self.service.close()


def build_service() -> ResearchService:
    state_dir = Path(os.environ.get("CODEX_RESEARCH_STATE_DIR", DEFAULT_STATE_DIR))
    startup_cwd = Path(
        os.environ.get("CODEX_RESEARCH_SOURCE_ROOT", os.getcwd())
    )
    max_concurrency = max(
        1, int(os.environ.get("CODEX_RESEARCH_MAX_CONCURRENCY", "4"))
    )
    command = os.environ.get("CODEX_RESEARCH_CODEX_BIN", "codex")
    timeout_seconds = float(
        os.environ.get("CODEX_RESEARCH_TIMEOUT_SECONDS", "3600")
    )
    return ResearchService(
        state_dir=state_dir,
        startup_cwd=startup_cwd,
        max_concurrency=max_concurrency,
        upstream_factory=lambda: CodexMcpClient(command, timeout_seconds),
    )


async def async_main() -> None:
    await McpServer(build_service()).run()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

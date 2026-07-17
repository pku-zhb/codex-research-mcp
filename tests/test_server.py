from __future__ import annotations

import asyncio
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest

from codex_research_mcp.server import (
    CodexMcpClient,
    ResearchService,
    TOOLS,
    UpstreamError,
)


THREAD_ID = "019f-research-test-thread"


class FakeUpstream:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.started = False

    @property
    def is_running(self) -> bool:
        return self.started

    async def start(self) -> None:
        self.started = True

    async def request(self, method: str, params: dict) -> dict:
        self.calls.append((method, params))
        return {
            "jsonrpc": "2.0",
            "id": len(self.calls),
            "result": {
                "content": [{"type": "text", "text": "evidence"}],
                "structuredContent": {
                    "threadId": THREAD_ID,
                    "content": "evidence",
                },
                "isError": False,
            },
        }

    async def close(self) -> None:
        self.started = False


class ResearchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.fake = FakeUpstream()
        self.service = ResearchService(
            state_dir=self.root / "state",
            startup_cwd=self.source,
            max_concurrency=2,
            upstream_factory=lambda: self.fake,
        )

    async def asyncTearDown(self) -> None:
        await self.service.close()
        self.temporary.cleanup()

    async def test_research_enforces_scratch_workspace(self) -> None:
        result = await self.service.research(
            {"prompt": "Find evidence", "source_cwd": str(self.source)}
        )
        self.assertEqual(
            result["structuredContent"]["threadId"], THREAD_ID
        )
        _, call = self.fake.calls[0]
        arguments = call["arguments"]
        scratch = Path(arguments["cwd"])
        self.assertTrue(scratch.is_dir())
        self.assertNotEqual(scratch, self.source)
        self.assertEqual(arguments["approval-policy"], "never")
        self.assertEqual(arguments["sandbox"], "workspace-write")
        self.assertTrue(
            arguments["config"]["sandbox_workspace_write.network_access"]
        )
        self.assertIn(str(self.source), arguments["developer-instructions"])
        self.assertIn(str(scratch), arguments["developer-instructions"])
        self.assertIn("extract, organize", arguments["developer-instructions"])
        self.assertIn("Do not take over", arguments["developer-instructions"])
        self.assertIn("lead researcher owns research direction", arguments["developer-instructions"])
        marker = self.service.thread_root / f"{THREAD_ID}.json"
        self.assertEqual(json.loads(marker.read_text())["scratchDir"], str(scratch))

    async def test_reply_requires_a_server_created_thread(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown research thread"):
            await self.service.research_reply(
                {"threadId": THREAD_ID, "prompt": "Continue"}
            )

    async def test_reply_uses_official_codex_reply_tool(self) -> None:
        await self.service.research({"prompt": "Find evidence"})
        await self.service.research_reply(
            {"threadId": THREAD_ID, "prompt": "Check conflict"}
        )
        _, call = self.fake.calls[1]
        self.assertEqual(call["name"], "codex-reply")
        self.assertEqual(call["arguments"]["threadId"], THREAD_ID)

    async def test_reply_thread_is_not_authorized_after_server_restart(self) -> None:
        await self.service.research({"prompt": "Find evidence"})
        restarted = ResearchService(
            state_dir=self.root / "state",
            startup_cwd=self.source,
            max_concurrency=2,
            upstream_factory=FakeUpstream,
        )
        try:
            with self.assertRaisesRegex(ValueError, "this MCP session"):
                await restarted.research_reply(
                    {"threadId": THREAD_ID, "prompt": "Continue"}
                )
        finally:
            await restarted.close()

    async def test_source_cwd_must_be_absolute(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute"):
            await self.service.research(
                {"prompt": "Find evidence", "source_cwd": "relative/path"}
            )


class CodexMcpClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.command = self.root / "fake-codex"
        self.command.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import sys

                for raw_line in sys.stdin:
                    message = json.loads(raw_line)
                    method = message.get("method")
                    message_id = message.get("id")
                    if method == "initialize":
                        result = {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "fake-codex", "version": "1"},
                        }
                    elif method == "tools/call":
                        notification = {
                            "jsonrpc": "2.0",
                            "method": "codex/event",
                            "params": {"payload": "x" * 70000},
                        }
                        print(json.dumps(notification, separators=(",", ":")), flush=True)
                        result = {
                            "content": [{"type": "text", "text": "evidence"}],
                            "structuredContent": {
                                "threadId": f"019f-research-thread-{message_id:04d}",
                                "content": "evidence",
                            },
                            "isError": False,
                        }
                    else:
                        continue
                    response = {"jsonrpc": "2.0", "id": message_id, "result": result}
                    print(json.dumps(response, separators=(",", ":")), flush=True)
                """
            ),
            encoding="utf-8",
        )
        os.chmod(self.command, 0o755)

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    async def test_concurrent_requests_accept_large_json_rpc_events(self) -> None:
        client = CodexMcpClient(str(self.command), timeout_seconds=5)
        service = ResearchService(
            state_dir=self.root / "state",
            startup_cwd=self.source,
            max_concurrency=3,
            upstream_factory=lambda: client,
        )
        try:
            results = await asyncio.gather(
                *(service.research({"prompt": f"Question {index}"}) for index in range(3))
            )
            self.assertEqual(
                [result["structuredContent"]["content"] for result in results],
                ["evidence", "evidence", "evidence"],
            )
            self.assertTrue(client.is_running)
        finally:
            await service.close()

    async def test_reader_failure_is_reported_and_next_call_restarts(self) -> None:
        limits = iter((1024, 128 * 1024))
        clients: list[CodexMcpClient] = []

        def make_client() -> CodexMcpClient:
            client = CodexMcpClient(
                str(self.command),
                timeout_seconds=5,
                stream_limit_bytes=next(limits),
            )
            clients.append(client)
            return client

        service = ResearchService(
            state_dir=self.root / "state",
            startup_cwd=self.source,
            max_concurrency=1,
            upstream_factory=make_client,
        )
        try:
            error_output = io.StringIO()
            with redirect_stderr(error_output):
                with self.assertRaisesRegex(UpstreamError, "configured 1024-byte"):
                    await service.research({"prompt": "Trigger reader failure"})
            self.assertIn("stdout reader failed", error_output.getvalue())
            await asyncio.sleep(0)
            self.assertFalse(clients[0].is_running)

            result = await service.research({"prompt": "Retry on a fresh reader"})
            self.assertEqual(result["structuredContent"]["content"], "evidence")
            self.assertEqual(len(clients), 2)
            self.assertTrue(clients[1].is_running)
        finally:
            await service.close()


class ToolSchemaTests(unittest.TestCase):
    def test_both_tools_are_marked_concurrency_safe(self) -> None:
        self.assertEqual([tool["name"] for tool in TOOLS], ["research", "research_reply"])
        self.assertIn("evidence assignment", TOOLS[0]["description"])
        for tool in TOOLS:
            self.assertIs(tool["annotations"]["readOnlyHint"], True)
            self.assertIs(tool["annotations"]["destructiveHint"], False)
            self.assertIs(tool["annotations"]["openWorldHint"], True)


if __name__ == "__main__":
    unittest.main()

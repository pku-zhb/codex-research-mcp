from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from codex_research_mcp.server import ResearchService, TOOLS


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


class ToolSchemaTests(unittest.TestCase):
    def test_both_tools_are_marked_concurrency_safe(self) -> None:
        self.assertEqual([tool["name"] for tool in TOOLS], ["research", "research_reply"])
        for tool in TOOLS:
            self.assertIs(tool["annotations"]["readOnlyHint"], True)
            self.assertIs(tool["annotations"]["destructiveHint"], False)
            self.assertIs(tool["annotations"]["openWorldHint"], True)


if __name__ == "__main__":
    unittest.main()

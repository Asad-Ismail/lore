from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def reload_lore(workspace_root: str) -> None:
    os.environ["LORE_REPO_ROOT"] = workspace_root
    for name in list(sys.modules):
        if name == "lore" or name.startswith("lore."):
            del sys.modules[name]


class DemoWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        reload_lore(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()
        os.environ.pop("LORE_REPO_ROOT", None)
        for name in list(sys.modules):
            if name == "lore" or name.startswith("lore."):
                del sys.modules[name]

    def test_seed_demo_populates_workspace(self) -> None:
        from lore.config import RAW_DIR, WIKI_DIR
        from lore.demo import seed_demo

        stats = seed_demo()

        self.assertGreaterEqual(stats["articles"], 5)
        self.assertTrue((RAW_DIR / "articles" / "llm-wiki.md").exists())
        self.assertTrue((WIKI_DIR / "concepts" / "LLM Wiki.md").exists())
        self.assertTrue((WIKI_DIR / "_index.md").exists())

    def test_ingest_demo_source_writes_summary_page(self) -> None:
        from lore.config import REPO_ROOT
        from lore.demo import ingest_demo_source, seed_demo

        seed_demo()

        note_path = Path(self.tempdir.name) / "fresh-note.md"
        note_path.write_text(
            "# Fresh Note\n\n"
            "A markdown wiki compounds memory across conversations.\n"
            "MCP exposes the same tools to multiple clients.\n"
            "Question traces keep the next suggestion tied to what the user actually asks.\n",
            encoding="utf-8",
        )

        result = ingest_demo_source(note_path, kind="file")
        article_path = REPO_ROOT / result.article_path

        self.assertEqual(result.action, "created")
        self.assertEqual(len(result.suggestions), 3)
        self.assertTrue(article_path.exists())

        content = article_path.read_text(encoding="utf-8")
        self.assertIn("[[LLM Wiki]]", content)
        self.assertIn("[[Model Context Protocol]]", content)


if __name__ == "__main__":
    unittest.main()

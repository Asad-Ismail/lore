from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lore.preview import build_preview_articles, render_article_markdown, render_graph_html


class PreviewGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.wiki_dir = Path(self.tempdir.name) / "wiki"
        (self.wiki_dir / "concepts").mkdir(parents=True)
        (self.wiki_dir / "meta").mkdir(parents=True)

        (self.wiki_dir / "concepts" / "LLM Wiki.md").write_text(
            "# LLM Wiki\n\n"
            "A durable markdown wiki for agent memory.\n\n"
            "## Connections\n\n"
            "- [[Model Context Protocol]]\n\n"
            "## Referenced By\n\n"
            "- [[Demo Tour]]\n",
            encoding="utf-8",
        )
        (self.wiki_dir / "concepts" / "Model Context Protocol.md").write_text(
            "# Model Context Protocol\n\n"
            "A stable tool surface for the same wiki.\n",
            encoding="utf-8",
        )
        (self.wiki_dir / "meta" / "Demo Tour.md").write_text(
            "# Demo Tour\n\n"
            "The preview of the Lore vault.\n\n"
            "## Connections\n\n"
            "- [[LLM Wiki]]\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_build_preview_articles_ignores_referenced_by_footer(self) -> None:
        articles = build_preview_articles(self.wiki_dir)
        by_title = {article.title: article for article in articles}

        self.assertEqual(by_title["LLM Wiki"].outgoing, ("concepts/Model Context Protocol.md",))
        self.assertEqual(by_title["Model Context Protocol"].incoming, ("concepts/LLM Wiki.md",))
        self.assertEqual(by_title["Demo Tour"].outgoing, ("concepts/LLM Wiki.md",))
        self.assertIn("meta/Demo Tour.md", by_title["LLM Wiki"].incoming)

    def test_renderers_surface_selected_page_and_graph_focus(self) -> None:
        articles = build_preview_articles(self.wiki_dir)

        article_md = render_article_markdown(articles, "concepts/LLM Wiki.md")
        graph_html = render_graph_html(articles, "concepts/LLM Wiki.md")

        self.assertIn("Links out: **1**", article_md)
        self.assertIn("Backlinks: `Demo Tour`", article_md)
        self.assertIn("Deterministic graph preview", graph_html)
        self.assertIn("Focus: LLM Wiki", graph_html)


if __name__ == "__main__":
    unittest.main()

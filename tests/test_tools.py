"""工具的离线测试：用假的下载函数代替网络。运行：python -m unittest discover tests"""
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from paper_agent.tools import PaperTools, ToolError, normalize_arxiv_id

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2308.04079v1</id>
    <published>2023-08-08T17:58:53Z</published>
    <title>3D Gaussian Splatting for Real-Time
      Radiance Field Rendering</title>
    <summary>  Radiance Field methods have recently revolutionized novel-view synthesis. </summary>
    <author><name>Bernhard Kerbl</name></author>
    <author><name>Georgios Kopanas</name></author>
    <arxiv:primary_category term="cs.GR"/>
  </entry>
</feed>"""


class FakeFetch:
    def __init__(self):
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        return FEED


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.fetch = FakeFetch()
        self.tools = PaperTools(root / "cache", root / "notes", fetch=self.fetch, min_interval=0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_normalize_id(self):
        self.assertEqual(normalize_arxiv_id("https://arxiv.org/abs/2308.04079v2"), "2308.04079v2")
        self.assertEqual(normalize_arxiv_id("arXiv:2308.04079"), "2308.04079")
        self.assertEqual(normalize_arxiv_id("hep-th/9901001"), "hep-th/9901001")
        with self.assertRaises(ToolError):
            normalize_arxiv_id("../../etc/passwd")

    def test_search_parses_feed(self):
        out = self.tools.search_arxiv("gaussian splatting")
        paper = out["results"][0]
        self.assertEqual(paper["arxiv_id"], "2308.04079")
        self.assertEqual(paper["title"], "3D Gaussian Splatting for Real-Time Radiance Field Rendering")
        self.assertEqual(paper["authors"], ["Bernhard Kerbl", "Georgios Kopanas"])
        self.assertEqual(paper["category"], "cs.GR")
        self.assertIn('ti:"gaussian splatting"', urllib.parse.unquote_plus(self.fetch.urls[0]))
        self.assertIn("all%3Agaussian+AND+all%3Asplatting", self.fetch.urls[1])

    def test_search_single_word_uses_one_request(self):
        self.tools.search_arxiv("GaussianEditor")
        self.assertEqual(len(self.fetch.urls), 1)

    def test_search_drops_stopwords_and_tries_title_phrase(self):
        out = self.tools.search_arxiv("Attention Is All You Need")
        self.assertEqual(len(out["results"]), 1)  # 两次请求返回同一篇，去重后只剩一条
        title_url, keyword_url = map(urllib.parse.unquote_plus, self.fetch.urls)
        self.assertIn('ti:"Attention Is All You Need"', title_url)
        self.assertIn("all:Attention AND all:All AND all:You AND all:Need", keyword_url)
        self.assertNotIn("all:Is", keyword_url)

    def test_search_keeps_raw_arxiv_syntax(self):
        self.tools.search_arxiv('ti:"segment anything"')
        self.assertEqual(len(self.fetch.urls), 1)
        self.assertIn('ti:"segment anything"', urllib.parse.unquote_plus(self.fetch.urls[0]))

    def test_read_and_find_use_cached_text(self):
        cache = Path(self.tmp.name) / "cache"
        cache.mkdir()
        (cache / "2308.04079.txt").write_text("abc " * 3000 + "PSNR 27.21 on Mip-NeRF360", encoding="utf-8")
        (cache / "2308.04079.meta.json").write_text('{"title": "3DGS"}', encoding="utf-8")

        first = self.tools.read_paper("2308.04079", max_chars=1000)
        self.assertEqual(first["next_offset"], 1000)
        self.assertEqual(first["metadata"]["title"], "3DGS")
        last = self.tools.read_paper("2308.04079", offset=first["total_chars"] - 10)
        self.assertIsNone(last["next_offset"])
        self.assertNotIn("metadata", last)

        found = self.tools.find_in_paper("2308.04079", "psnr", window=50)
        self.assertEqual(found["hit_count"], 1)
        self.assertIn("27.21", found["hits"][0]["snippet"])
        self.assertEqual(self.fetch.urls, [])  # 全部走缓存，没有联网

    def test_save_note_stays_in_notes_dir(self):
        out = self.tools.save_note("../../evil", "# 笔记")
        self.assertEqual(Path(out["saved"]), Path(self.tmp.name) / "notes" / "evil.md")
        with self.assertRaises(ToolError):
            self.tools.save_note(".hidden.md", "x")

    def test_dispatch_turns_errors_into_json(self):
        self.assertIn("error", json.loads(self.tools.dispatch("no_such_tool", {})))
        self.assertIn("参数不对", json.loads(self.tools.dispatch("search_arxiv", {"q": "x"}))["error"])
        self.assertIn("不是合法", json.loads(self.tools.dispatch("read_paper", {"arxiv_id": "hello"}))["error"])


if __name__ == "__main__":
    unittest.main()

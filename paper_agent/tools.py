"""Agent 能调用的 4 个工具：搜论文、分段读全文、在全文里找关键词、保存笔记。

每个工具都返回 JSON 字符串交给模型；出错时返回 {"error": ...}，
让模型自己决定换个参数重试，而不是让整个程序崩掉。
"""
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{}"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "paper-agent/0.1 (learning project)"
# 新式编号 2308.04079(v2)，旧式编号 hep-th/9901001
ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7})(v\d+)?$")
MAX_PDF_BYTES = 40 * 1024 * 1024
MAX_READ_CHARS = 12000
# arXiv 不索引这些词，查询里只要出现一个，AND 连起来就一条都搜不到（评测中发现）
STOPWORDS = {"a", "an", "the", "for", "of", "and", "or", "in", "on", "to", "is", "are", "with",
             "by", "from", "as", "at", "via", "be", "its", "into", "we", "our", "this", "that"}


class ToolError(Exception):
    pass


def _http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise ToolError("文件超过 40MB，放弃下载")
    return data


def _clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_arxiv_id(raw):
    """接受 '2308.04079'、'arXiv:2308.04079v2'、'https://arxiv.org/abs/2308.04079' 等写法。"""
    s = str(raw).strip()
    s = re.sub(r"^https?://(www\.)?arxiv\.org/(abs|pdf)/", "", s)
    s = re.sub(r"^arxiv:", "", s, flags=re.IGNORECASE)
    s = s.removesuffix(".pdf")
    if not ARXIV_ID_RE.match(s):
        raise ToolError(f"不是合法的 arXiv 编号：{raw}")
    return s


class PaperTools:
    def __init__(self, cache_dir, notes_dir, fetch=_http_get, min_interval=3.0):
        self.cache_dir = Path(cache_dir)
        self.notes_dir = Path(notes_dir)
        self.fetch = fetch  # 测试时换成假的下载函数，就能离线跑
        self.min_interval = min_interval  # arXiv 要求两次请求间隔约 3 秒
        self._last_request = 0.0

    # ---------- 对模型暴露的工具 ----------

    def search_arxiv(self, query, max_results=5):
        query = _clean(query)
        if not query:
            raise ToolError("query 不能为空")
        max_results = max(1, min(int(max_results), 10))
        if ":" in query:  # 含冒号说明模型自己写了 arXiv 检索语法（如 ti:xxx），原样使用
            searches = [query]
        else:
            words = [w for w in re.split(r"[^\w\-]+", query) if w]
            terms = [w for w in words if w.lower() not in STOPWORDS] or words
            searches = [" AND ".join(f"all:{w}" for w in terms)]
            if len(words) >= 2:  # 用户常给的是标题，先按标题短语精确搜，再补关键词结果
                searches.insert(0, 'ti:"' + " ".join(words) + '"')
        results, seen = [], set()
        for search in searches:
            params = urllib.parse.urlencode({"search_query": search, "max_results": max_results, "sortBy": "relevance"})
            for paper in self._parse_feed(self._get(f"{ARXIV_API}?{params}")):
                if paper["arxiv_id"] not in seen:
                    seen.add(paper["arxiv_id"])
                    results.append(paper)
        return {"query": query, "results": results[:max_results]}

    def read_paper(self, arxiv_id, offset=0, max_chars=6000):
        arxiv_id = normalize_arxiv_id(arxiv_id)
        text = self._paper_text(arxiv_id)
        offset = max(0, int(offset))
        max_chars = max(500, min(int(max_chars), MAX_READ_CHARS))
        end = min(offset + max_chars, len(text))
        result = {
            "arxiv_id": arxiv_id,
            "total_chars": len(text),
            "offset": offset,
            "next_offset": end if end < len(text) else None,  # None 表示已经读到结尾
            "text": text[offset:end],
        }
        if offset == 0:
            result["metadata"] = self._metadata(arxiv_id)
        return result

    def find_in_paper(self, arxiv_id, keyword, window=300, max_hits=5):
        arxiv_id = normalize_arxiv_id(arxiv_id)
        keyword = str(keyword).strip()
        if not keyword:
            raise ToolError("keyword 不能为空")
        text = self._paper_text(arxiv_id)
        window = max(50, min(int(window), 1000))
        max_hits = max(1, min(int(max_hits), 10))
        hits = []
        for m in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
            start = max(0, m.start() - window)
            hits.append({"offset": start, "snippet": text[start:m.end() + window]})
            if len(hits) >= max_hits:
                break
        return {"arxiv_id": arxiv_id, "keyword": keyword, "hit_count": len(hits), "hits": hits}

    def save_note(self, filename, content):
        name = Path(str(filename)).name  # 只取文件名，防止 ../../ 写到 notes 目录外面
        if not name or name.startswith("."):
            raise ToolError(f"文件名不合法：{filename}")
        if not name.endswith(".md"):
            name += ".md"
        if not str(content).strip():
            raise ToolError("笔记内容为空")
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        path = self.notes_dir / name
        path.write_text(str(content), encoding="utf-8")
        return {"saved": str(path), "chars": len(str(content))}

    def dispatch(self, name, args):
        """按名字调用工具，结果统一转成 JSON 字符串。"""
        func = {
            "search_arxiv": self.search_arxiv,
            "read_paper": self.read_paper,
            "find_in_paper": self.find_in_paper,
            "save_note": self.save_note,
        }.get(name)
        if func is None:
            return json.dumps({"error": f"没有叫 {name} 的工具"}, ensure_ascii=False)
        try:
            return json.dumps(func(**args), ensure_ascii=False)
        except TypeError as e:  # 模型传了不存在或缺少的参数
            return json.dumps({"error": f"参数不对：{e}"}, ensure_ascii=False)
        except ToolError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        except Exception as e:  # 网络、解析等意外错误也交给模型处理
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

    # ---------- 内部辅助 ----------

    def _get(self, url):
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.fetch(url)
        finally:
            self._last_request = time.monotonic()

    def _parse_feed(self, xml_bytes):
        root = ET.fromstring(xml_bytes)
        papers = []
        for entry in root.findall("a:entry", ATOM):
            abs_url = entry.findtext("a:id", default="", namespaces=ATOM)
            if "/abs/" not in abs_url:  # 查不到时 arXiv 会返回一条错误 entry
                continue
            authors = [_clean(a.findtext("a:name", default="", namespaces=ATOM)) for a in entry.findall("a:author", ATOM)]
            primary = entry.find("arxiv:primary_category", ATOM)
            papers.append({
                "arxiv_id": re.sub(r"v\d+$", "", abs_url.split("/abs/")[-1]),
                "title": _clean(entry.findtext("a:title", default="", namespaces=ATOM)),
                "authors": authors[:6] + (["等"] if len(authors) > 6 else []),
                "published": entry.findtext("a:published", default="", namespaces=ATOM)[:10],
                "category": primary.get("term") if primary is not None else "",
                "abstract": _clean(entry.findtext("a:summary", default="", namespaces=ATOM))[:800],
            })
        return papers

    def _metadata(self, arxiv_id):
        path = self.cache_dir / f"{arxiv_id.replace('/', '_')}.meta.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        params = urllib.parse.urlencode({"id_list": arxiv_id})
        papers = self._parse_feed(self._get(f"{ARXIV_API}?{params}"))
        meta = papers[0] if papers else {}
        meta.pop("abstract", None)  # 正文开头已经有摘要，不重复占上下文
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return meta

    def _paper_text(self, arxiv_id):
        """下载 PDF 并转成纯文本，结果缓存到本地，同一篇论文只下载一次。"""
        stem = arxiv_id.replace("/", "_")
        txt_path = self.cache_dir / f"{stem}.txt"
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = self.cache_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            data = self._get(ARXIV_PDF.format(arxiv_id))
            if not data.startswith(b"%PDF"):
                raise ToolError(f"{arxiv_id} 下载到的不是 PDF，可能编号不存在")
            pdf_path.write_bytes(data)
        try:
            proc = subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
                                  capture_output=True, timeout=120)
        except FileNotFoundError as e:
            raise ToolError("系统里没有 pdftotext，请先安装 poppler") from e
        if proc.returncode != 0:
            raise ToolError(f"PDF 转文字失败：{proc.stderr.decode(errors='replace')[:200]}")
        text = re.sub(r"\n{3,}", "\n\n", proc.stdout.decode("utf-8", errors="replace"))
        txt_path.write_text(text, encoding="utf-8")
        return text


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "在 arXiv 上按关键词搜索论文，返回编号、标题、作者、日期和摘要。关键词用英文效果最好。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "英文关键词，例如 'gaussian splatting real-time rendering'；也可以用 arXiv 语法如 'ti:\"segment anything\"'"},
                    "max_results": {"type": "integer", "description": "返回条数，1–10，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": "分段读取论文全文（PDF 转成的文字）。第一次从 offset=0 读，会附带元数据；返回的 next_offset 用于读下一段，为 null 表示已读完。",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv 编号，如 2308.04079"},
                    "offset": {"type": "integer", "description": "从第几个字符开始读，默认 0"},
                    "max_chars": {"type": "integer", "description": "本次最多读多少字符，500–12000，默认 6000"},
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_in_paper",
            "description": "在论文全文里查找关键词（不区分大小写），返回附近的原文片段和位置。适合定位实验数字、数据集名、表格等，比从头读更省。",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv 编号"},
                    "keyword": {"type": "string", "description": "要找的词，如 'PSNR'、'Table 1'、'ScanNet'"},
                    "window": {"type": "integer", "description": "命中位置前后各取多少字符，默认 300"},
                    "max_hits": {"type": "integer", "description": "最多返回几处，默认 5"},
                },
                "required": ["arxiv_id", "keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "把 Markdown 笔记保存到本地 notes 目录。只有用户要求保存笔记时才调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名，如 3dgs.md"},
                    "content": {"type": "string", "description": "完整的 Markdown 笔记内容"},
                },
                "required": ["filename", "content"],
            },
        },
    },
]

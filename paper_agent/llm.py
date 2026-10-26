"""DeepSeek 对话接口的最小封装。

只用标准库 urllib 直接发 HTTP 请求，不依赖 openai SDK，
这样每一步（请求体、工具定义、重试）都看得见。
DeepSeek 的接口格式与 OpenAI Chat Completions 兼容。
"""
import json
import os
import time
import urllib.error
import urllib.request

RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


class DeepSeekClient:
    def __init__(self, api_key=None, model=None, base_url=None, timeout=120, max_retries=3):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise LLMError("没有找到 DEEPSEEK_API_KEY：请在项目根目录的 .env 文件里配置（参考 .env.example）")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages, tools=None):
        """发一轮对话，返回 {"message": 模型回复, "usage": token 用量}。"""
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return {"message": data["choices"][0]["message"], "usage": data.get("usage", {})}
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:500]
                # 限流和服务端错误可以重试；401/400 这类是请求本身有问题，重试没用
                if e.code in RETRY_STATUS and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(f"DeepSeek 返回 HTTP {e.code}：{detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(f"连不上 DeepSeek：{e}") from e

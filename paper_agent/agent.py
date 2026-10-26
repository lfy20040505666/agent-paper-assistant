"""Agent 主循环：模型思考 → 调工具 → 看结果 → 再思考，直到给出最终回答。

没有用 LangChain 之类的框架，整个循环就是下面 run() 里的一个 for 循环。
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .tools import TOOL_SCHEMAS

SYSTEM_PROMPT = """你是一个论文速读助手，用中文回答。你可以调用工具搜索 arXiv、读取论文全文、在全文里查关键词、保存笔记。

工作规则：
1. 涉及具体论文的事实（编号、作者、数字、数据集），必须先用工具查到原文再回答，不要凭记忆编。
2. 找数字或某个数据集时，优先用 find_in_paper 定位，再用 read_paper 读附近内容，不要从头读整篇。
3. 回答里写明依据：论文的 arXiv 编号，以及数字出自哪张表或哪一节。
4. 查不到就直说“没有在论文中找到”，不要猜。同名论文有多篇时，把每一篇都列出来。
5. 只有用户要求保存笔记时才调用 save_note。速读笔记包含：一句话总结、要解决的问题、核心方法、主要实验结果、局限。
"""

FINAL_NUDGE = "已经达到步数上限，不能再调用工具。请根据目前查到的信息直接给出最终回答，并说明哪些内容没有查到。"


@dataclass
class AgentResult:
    answer: str
    stopped_reason: str  # "final"：模型主动结束；"max_steps"：步数用完被要求收尾
    steps: int
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    trace_path: str = ""


class PaperAgent:
    def __init__(self, llm, tools, max_steps=12, max_tool_output=12000, trace_dir=None, verbose=False):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.max_tool_output = max_tool_output  # 单次工具结果的上限，防止一次塞爆上下文
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.verbose = verbose

    def run(self, task):
        start = time.monotonic()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
        result = AgentResult(answer="", stopped_reason="", steps=0)
        trace = self._open_trace()
        self._log(trace, {"type": "task", "task": task})

        for step in range(1, self.max_steps + 1):
            result.steps = step
            msg = self._ask(messages, result, trace, tools=TOOL_SCHEMAS)
            calls = msg.get("tool_calls") or []
            if not calls:  # 模型不再调工具，说明它认为可以回答了
                result.answer = msg.get("content") or ""
                result.stopped_reason = "final"
                break
            for call in calls:
                output = self._run_tool(call, result)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                self._log(trace, {"type": "tool", "step": step, "name": call["function"]["name"],
                                  "arguments": call["function"].get("arguments"), "output": output[:2000]})
        else:
            # 步数用完还没结束：不给工具，逼模型用已有信息收尾，而不是直接报错
            messages.append({"role": "user", "content": FINAL_NUDGE})
            msg = self._ask(messages, result, trace, tools=None)
            result.answer = msg.get("content") or ""
            result.stopped_reason = "max_steps"

        result.seconds = round(time.monotonic() - start, 1)
        self._log(trace, {"type": "result", "answer": result.answer, "stopped_reason": result.stopped_reason,
                          "steps": result.steps, "tool_calls": result.tool_calls,
                          "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                          "seconds": result.seconds})
        if trace:
            trace.close()
            result.trace_path = trace.name
        return result

    def _ask(self, messages, result, trace, tools):
        resp = self.llm.chat(messages, tools=tools)
        msg, usage = resp["message"], resp.get("usage", {})
        result.prompt_tokens += usage.get("prompt_tokens", 0)
        result.completion_tokens += usage.get("completion_tokens", 0)
        # 只保留下一轮需要的字段（例如丢掉推理模型的 reasoning_content，它不能回传）
        kept = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            kept["tool_calls"] = msg["tool_calls"]
        messages.append(kept)
        self._log(trace, {"type": "llm", "step": result.steps, "content": kept["content"],
                          "tool_calls": [c["function"] for c in msg.get("tool_calls") or []], "usage": usage})
        if self.verbose and kept["content"] and msg.get("tool_calls"):
            print(f"  [思考] {kept['content'][:200]}")
        return msg

    def _run_tool(self, call, result):
        name = call["function"]["name"]
        raw_args = call["function"].get("arguments") or "{}"
        result.tool_calls.append(name)
        try:
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                raise ValueError("参数必须是 JSON 对象")
        except ValueError as e:  # json.JSONDecodeError 也是 ValueError
            output = json.dumps({"error": f"参数不是合法 JSON：{e}"}, ensure_ascii=False)
        else:
            if self.verbose:
                print(f"  [工具] {name}({json.dumps(args, ensure_ascii=False)[:150]})")
            output = self.tools.dispatch(name, args)
        if len(output) > self.max_tool_output:
            output = output[:self.max_tool_output] + "…（结果过长已截断，请缩小范围或分段读取）"
        return output

    def _open_trace(self):
        if not self.trace_dir:
            return None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl"
        return open(self.trace_dir / name, "w", encoding="utf-8")

    @staticmethod
    def _log(trace, event):
        if trace:
            trace.write(json.dumps(event, ensure_ascii=False) + "\n")
            trace.flush()

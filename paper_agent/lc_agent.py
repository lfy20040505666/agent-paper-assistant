"""LangChain 版 Agent：和手写版（agent.py）用同一套工具、同一段系统提示，方便直接对比。

需要额外安装依赖：pip install -r requirements-langchain.txt
Agent 循环由 langchain 的 create_agent（底层是 LangGraph 状态图）负责，
这里只做三件事：把 PaperTools 包装成 LangChain 工具、配置模型、把结果整理成和手写版一样的 AgentResult。
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from .agent import SYSTEM_PROMPT, AgentResult
from .tools import TOOL_SCHEMAS


def make_deepseek_model():
    """DeepSeek 兼容 OpenAI 接口，所以直接用 ChatOpenAI 换个 base_url。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY：请在项目根目录的 .env 文件里配置（参考 .env.example）")
    return ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=api_key,
        temperature=0,
        max_retries=3,
        timeout=120,
    )


def make_lc_tools(paper_tools, max_tool_output=12000):
    """把 4 个工具包装成 LangChain 的 StructuredTool，参数说明直接复用 TOOL_SCHEMAS。"""
    lc_tools = []
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]

        def run(_name=fn["name"], **kwargs):
            # 和手写版一样走 dispatch：出错时返回 {"error": ...} 而不是抛异常
            output = paper_tools.dispatch(_name, kwargs)
            if len(output) > max_tool_output:
                output = output[:max_tool_output] + "…（结果过长已截断，请缩小范围或分段读取）"
            return output

        lc_tools.append(StructuredTool.from_function(
            func=run, name=fn["name"], description=fn["description"], args_schema=fn["parameters"]))
    return lc_tools


class LangChainPaperAgent:
    def __init__(self, model, paper_tools, max_steps=12, max_tool_output=12000, trace_dir=None, verbose=False):
        self.max_steps = max_steps
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.verbose = verbose
        self.graph = create_agent(
            model,
            tools=make_lc_tools(paper_tools, max_tool_output),
            system_prompt=SYSTEM_PROMPT,
            # 最多调用模型 max_steps 次，超过就直接结束（不像手写版那样再给一次收尾机会）
            middleware=[ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="end")],
        )

    def run(self, task):
        start = time.monotonic()
        # recursion_limit 是 LangGraph 的节点执行次数上限，设得比步数上限宽松，让 middleware 先生效
        state = self.graph.invoke({"messages": [{"role": "user", "content": task}]},
                                  config={"recursion_limit": self.max_steps * 4 + 10})
        messages = state["messages"]

        result = AgentResult(answer="", stopped_reason="final", steps=0)
        model_replies = [m for m in messages if isinstance(m, AIMessage) and m.response_metadata]
        result.steps = len(model_replies)
        for m in model_replies:
            usage = m.usage_metadata or {}
            result.prompt_tokens += usage.get("input_tokens", 0)
            result.completion_tokens += usage.get("output_tokens", 0)
        for m in messages:
            if isinstance(m, AIMessage):
                result.tool_calls += [c["name"] for c in m.tool_calls]

        last = messages[-1]
        result.answer = last.content if isinstance(last.content, str) else str(last.content)
        # middleware 超限时插入的那条消息不是模型生成的，没有 response_metadata
        if isinstance(last, AIMessage) and not last.response_metadata:
            result.stopped_reason = "max_steps"
        result.seconds = round(time.monotonic() - start, 1)
        result.trace_path = self._write_trace(task, messages, result)
        if self.verbose:
            for name in result.tool_calls:
                print(f"  [工具] {name}")
        return result

    def _write_trace(self, task, messages, result):
        """写成和手写版相同格式的 JSONL，评测脚本可以用同一套方式统计。"""
        if not self.trace_dir:
            return ""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-langchain.jsonl")
        events = [{"type": "task", "task": task, "impl": "langchain"}]
        step = 0
        for m in messages:
            if isinstance(m, AIMessage):
                step += 1
                events.append({"type": "llm", "step": step, "content": m.content,
                               "tool_calls": [{"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}
                                              for c in m.tool_calls],
                               "usage": m.usage_metadata or {}})
            elif isinstance(m, ToolMessage):
                args = next((c["args"] for prev in messages if isinstance(prev, AIMessage)
                             for c in prev.tool_calls if c["id"] == m.tool_call_id), {})
                events.append({"type": "tool", "step": step, "name": m.name,
                               "arguments": json.dumps(args, ensure_ascii=False), "output": str(m.content)[:2000]})
        events.append({"type": "result", "answer": result.answer, "stopped_reason": result.stopped_reason,
                       "steps": result.steps, "tool_calls": result.tool_calls,
                       "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                       "seconds": result.seconds})
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
        return str(path)

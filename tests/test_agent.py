"""Agent 循环的离线测试：用按剧本回复的假模型代替 DeepSeek。"""
import json
import tempfile
import unittest
from pathlib import Path

from paper_agent.agent import FINAL_NUDGE, PaperAgent


def tool_call(name, args, call_id="call_1", raw=None):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args)}}


class ScriptedLLM:
    """依次返回预先写好的回复，并记下每次收到的 messages。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return {"message": self.replies.pop(0), "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class FakeTools:
    def __init__(self):
        self.calls = []

    def dispatch(self, name, args):
        self.calls.append((name, args))
        return json.dumps({"ok": True, "name": name})


class AgentTests(unittest.TestCase):
    def test_tool_then_final_answer(self):
        llm = ScriptedLLM([
            {"role": "assistant", "content": "先搜一下", "tool_calls": [tool_call("search_arxiv", {"query": "3dgs"})]},
            {"role": "assistant", "content": "编号是 2308.04079"},
        ])
        tools = FakeTools()
        result = PaperAgent(llm, tools).run("3DGS 的编号？")

        self.assertEqual(result.answer, "编号是 2308.04079")
        self.assertEqual(result.stopped_reason, "final")
        self.assertEqual(result.tool_calls, ["search_arxiv"])
        self.assertEqual(tools.calls, [("search_arxiv", {"query": "3dgs"})])
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (20, 10))
        # 第二轮请求里必须带上工具结果，并且 tool_call_id 对得上
        tool_msg = llm.calls[1]["messages"][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_1")

    def test_bad_json_arguments_go_back_to_model(self):
        llm = ScriptedLLM([
            {"role": "assistant", "content": "", "tool_calls": [tool_call("read_paper", None, raw="{oops")]},
            {"role": "assistant", "content": "参数写错了，抱歉"},
        ])
        tools = FakeTools()
        result = PaperAgent(llm, tools).run("读论文")

        self.assertEqual(tools.calls, [])  # 参数坏了就不该真的调用工具
        self.assertIn("不是合法 JSON", llm.calls[1]["messages"][-1]["content"])
        self.assertEqual(result.stopped_reason, "final")

    def test_max_steps_forces_answer_without_tools(self):
        loop = {"role": "assistant", "content": "", "tool_calls": [tool_call("search_arxiv", {"query": "x"})]}
        llm = ScriptedLLM([loop, loop, {"role": "assistant", "content": "只查到一部分"}])
        result = PaperAgent(llm, FakeTools(), max_steps=2).run("问题")

        self.assertEqual(result.stopped_reason, "max_steps")
        self.assertEqual(result.answer, "只查到一部分")
        self.assertIsNone(llm.calls[-1]["tools"])
        self.assertEqual(llm.calls[-1]["messages"][-1]["content"], FINAL_NUDGE)

    def test_long_tool_output_is_truncated_and_trace_written(self):
        class BigTools(FakeTools):
            def dispatch(self, name, args):
                return "x" * 50000

        llm = ScriptedLLM([
            {"role": "assistant", "content": "", "tool_calls": [tool_call("read_paper", {"arxiv_id": "2308.04079"})]},
            {"role": "assistant", "content": "好了", "reasoning_content": "不应回传"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            result = PaperAgent(llm, BigTools(), max_tool_output=1000, trace_dir=tmp).run("读")
            events = [json.loads(line) for line in Path(result.trace_path).read_text(encoding="utf-8").splitlines()]

        tool_content = llm.calls[1]["messages"][-1]["content"]
        self.assertLess(len(tool_content), 1100)
        self.assertIn("已截断", tool_content)
        self.assertEqual([e["type"] for e in events], ["task", "llm", "tool", "llm", "result"])

    def test_reasoning_content_not_sent_back(self):
        llm = ScriptedLLM([
            {"role": "assistant", "content": "", "reasoning_content": "内部推理",
             "tool_calls": [tool_call("search_arxiv", {"query": "x"})]},
            {"role": "assistant", "content": "完成"},
        ])
        PaperAgent(llm, FakeTools()).run("问题")
        assistant_msg = llm.calls[1]["messages"][2]
        self.assertNotIn("reasoning_content", assistant_msg)


if __name__ == "__main__":
    unittest.main()

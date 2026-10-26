"""LangChain 版的离线测试：用按剧本回复的假聊天模型。没装 LangChain 时自动跳过。"""
import json
import tempfile
import unittest
from pathlib import Path

try:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from paper_agent.lc_agent import LangChainPaperAgent
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


if HAS_LANGCHAIN:
    class ScriptedChatModel(FakeMessagesListChatModel):
        """假模型本身不支持 bind_tools，这里让它忽略工具定义，按剧本依次回复。"""

        def bind_tools(self, tools, **kwargs):
            return self

    def reply(content="", tool_calls=None, tokens=(10, 5)):
        return AIMessage(content=content, tool_calls=tool_calls or [],
                         response_metadata={"model_name": "fake"},
                         usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                                         "total_tokens": sum(tokens)})


class FakeTools:
    def __init__(self, output=None):
        self.calls = []
        self.output = output

    def dispatch(self, name, args):
        self.calls.append((name, args))
        return self.output or json.dumps({"ok": True, "name": name})


@unittest.skipUnless(HAS_LANGCHAIN, "没有安装 LangChain")
class LangChainAgentTests(unittest.TestCase):
    def test_tool_then_final_answer(self):
        model = ScriptedChatModel(responses=[
            reply(tool_calls=[{"name": "search_arxiv", "args": {"query": "3dgs"}, "id": "call_1"}]),
            reply("编号是 2308.04079"),
        ])
        tools = FakeTools()
        with tempfile.TemporaryDirectory() as tmp:
            result = LangChainPaperAgent(model, tools, trace_dir=tmp).run("3DGS 的编号？")
            events = [json.loads(line) for line in Path(result.trace_path).read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result.answer, "编号是 2308.04079")
        self.assertEqual(result.stopped_reason, "final")
        self.assertEqual(result.tool_calls, ["search_arxiv"])
        self.assertEqual(tools.calls, [("search_arxiv", {"query": "3dgs"})])
        self.assertEqual((result.steps, result.prompt_tokens, result.completion_tokens), (2, 20, 10))
        self.assertEqual([e["type"] for e in events], ["task", "llm", "tool", "llm", "result"])
        self.assertEqual(events[2]["name"], "search_arxiv")

    def test_long_tool_output_is_truncated(self):
        model = ScriptedChatModel(responses=[
            reply(tool_calls=[{"name": "read_paper", "args": {"arxiv_id": "2308.04079"}, "id": "call_1"}]),
            reply("好了"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            result = LangChainPaperAgent(model, FakeTools(output="x" * 50000), max_tool_output=1000,
                                         trace_dir=tmp).run("读")
            tool_event = [json.loads(line) for line in Path(result.trace_path).read_text(encoding="utf-8").splitlines()][2]
        self.assertIn("已截断", tool_event["output"])

    def test_model_call_limit_stops_run(self):
        # 每轮必须是不同的消息对象：LangGraph 按消息 id 合并，重复用同一个对象会被当成同一条
        model = ScriptedChatModel(responses=[
            reply(tool_calls=[{"name": "search_arxiv", "args": {"query": "x"}, "id": f"call_{i}"}]) for i in range(4)])
        tools = FakeTools()
        result = LangChainPaperAgent(model, tools, max_steps=2).run("问题")
        self.assertEqual(result.stopped_reason, "max_steps")
        self.assertEqual(result.steps, 2)


if __name__ == "__main__":
    unittest.main()

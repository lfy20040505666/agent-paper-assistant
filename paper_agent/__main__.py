"""命令行入口：python -m paper_agent "你的问题" """
import argparse
import sys

from .agent import PaperAgent
from .config import CACHE_DIR, NOTES_DIR, TRACE_DIR, load_dotenv
from .llm import DeepSeekClient, LLMError
from .tools import PaperTools


def main():
    parser = argparse.ArgumentParser(description="论文速读 Agent")
    parser.add_argument("task", help="要问的问题或要做的事，例如：找到 3DGS 原论文并写一份速读笔记")
    parser.add_argument("--max-steps", type=int, default=12, help="最多和模型来回几轮，默认 12")
    parser.add_argument("--quiet", action="store_true", help="不打印中间的工具调用")
    args = parser.parse_args()

    load_dotenv()
    try:
        llm = DeepSeekClient()
    except LLMError as e:
        sys.exit(str(e))
    agent = PaperAgent(llm, PaperTools(CACHE_DIR, NOTES_DIR), max_steps=args.max_steps,
                       trace_dir=TRACE_DIR, verbose=not args.quiet)
    try:
        result = agent.run(args.task)
    except LLMError as e:
        sys.exit(str(e))

    print("\n" + result.answer)
    print(f"\n---\n结束方式：{result.stopped_reason} | 轮数：{result.steps} | 工具调用：{len(result.tool_calls)} 次"
          f" | token：输入 {result.prompt_tokens} / 输出 {result.completion_tokens} | 用时 {result.seconds}s")
    print(f"逐步记录：{result.trace_path}")


if __name__ == "__main__":
    main()

"""跑评测题，统计成功率、工具调用次数、token 和耗时。

用法（在项目根目录）：
    python eval/run_eval.py              # 跑全部题
    python eval/run_eval.py --only find_3dgs number_3dgs_psnr
结果写到 eval/results/<时间>/ 下：summary.md（表格）和 results.json（明细）。
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper_agent.agent import PaperAgent  # noqa: E402
from paper_agent.config import CACHE_DIR, load_dotenv  # noqa: E402
from paper_agent.llm import DeepSeekClient, LLMError  # noqa: E402
from paper_agent.tools import PaperTools  # noqa: E402


def check(task_check, answer, notes_dir):
    """按题目里的规则判对错，返回 (是否通过, 原因)。"""
    low = answer.lower()
    for s in task_check.get("answer_contains_all", []):
        if s.lower() not in low:
            return False, f"回答里没有 {s}"
    if "answer_contains_any" in task_check and not any(s.lower() in low for s in task_check["answer_contains_any"]):
        return False, f"回答里没有 {task_check['answer_contains_any']} 中的任何一个"
    if "answer_regex" in task_check and not re.search(task_check["answer_regex"], answer, flags=re.IGNORECASE):
        return False, "回答不符合正则"
    if "note_file" in task_check:
        path = Path(notes_dir) / task_check["note_file"]
        if not path.exists():
            return False, f"没有保存笔记 {task_check['note_file']}"
        note = path.read_text(encoding="utf-8")
        if len(note) < task_check.get("note_min_chars", 0):
            return False, f"笔记只有 {len(note)} 字符"
        for s in task_check.get("note_contains_all", []):
            if s.lower() not in note.lower():
                return False, f"笔记里没有 {s}"
    return True, ""


def count_empty_searches(trace_path):
    """数一次运行里有几次 search_arxiv 一条结果都没返回，这类调用纯属浪费。"""
    if not trace_path:
        return 0
    count = 0
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["type"] == "tool" and event["name"] == "search_arxiv" and '"results": []' in event["output"]:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="只跑这些 id 的题")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--impl", choices=["handwritten", "langchain"], default="handwritten",
                        help="评测哪个版本：手写版（默认）或 LangChain 版")
    args = parser.parse_args()

    load_dotenv()
    try:
        llm = DeepSeekClient()  # 两个版本都读同一组环境变量，这里顺便检查 Key 并拿到模型名
    except LLMError as e:
        sys.exit(str(e))

    tasks = json.loads((ROOT / "eval" / "tasks.json").read_text(encoding="utf-8"))
    if args.only:
        tasks = [t for t in tasks if t["id"] in args.only]
    out_dir = ROOT / "eval" / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{args.impl}")
    notes_dir = out_dir / "notes"  # 每次评测单独存笔记，不和上次的混在一起
    # 论文缓存共用，避免重复下载；它不影响判分
    tools = PaperTools(CACHE_DIR, notes_dir)
    if args.impl == "langchain":
        from paper_agent.lc_agent import LangChainPaperAgent, make_deepseek_model
        agent = LangChainPaperAgent(make_deepseek_model(), tools, max_steps=args.max_steps,
                                    trace_dir=out_dir / "traces")
    else:
        agent = PaperAgent(llm, tools, max_steps=args.max_steps, trace_dir=out_dir / "traces")

    rows = []
    for i, t in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {t['id']} ...", flush=True)
        try:
            r = agent.run(t["task"])
            ok, reason = check(t["check"], r.answer, notes_dir)
            row = {"id": t["id"], "passed": ok, "reason": reason, "stopped_reason": r.stopped_reason,
                   "steps": r.steps, "tool_calls": len(r.tool_calls), "tools_used": r.tool_calls,
                   "empty_searches": count_empty_searches(r.trace_path),
                   "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                   "seconds": r.seconds, "answer": r.answer, "trace": r.trace_path}
        except Exception as e:  # 手写版抛 LLMError，LangChain 版抛 openai 的异常，统一记为失败
            row = {"id": t["id"], "passed": False, "reason": f"调用模型失败：{type(e).__name__}: {e}"[:300],
                   "stopped_reason": "error",
                   "steps": 0, "tool_calls": 0, "tools_used": [], "empty_searches": 0,
                   "prompt_tokens": 0, "completion_tokens": 0,
                   "seconds": 0, "answer": "", "trace": ""}
        rows.append(row)
        print(f"    {'通过' if row['passed'] else '失败 ' + row['reason']}  "
              f"（{row['tool_calls']} 次工具，{row['seconds']}s）", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = render_summary(rows, f"{llm.model}，{args.impl}")
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"明细：{out_dir / 'results.json'}")


def render_summary(rows, model):
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    avg = lambda key: sum(r[key] for r in rows) / n if n else 0  # noqa: E731
    lines = [
        f"# 评测结果（模型 {model}）",
        "",
        f"- 通过：{passed}/{n}（{passed / n:.0%}）" if n else "- 没有题目",
        f"- 平均工具调用：{avg('tool_calls'):.1f} 次；平均轮数：{avg('steps'):.1f}；"
        f"空结果搜索共 {sum(r['empty_searches'] for r in rows)} 次",
        f"- 平均 token：输入 {avg('prompt_tokens'):.0f} / 输出 {avg('completion_tokens'):.0f}",
        f"- 平均耗时：{avg('seconds'):.1f}s（含首次下载 PDF 的时间）",
        "",
        "| 题目 | 结果 | 工具调用 | 轮数 | 输入/输出 token | 耗时 | 失败原因 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {r['tool_calls']} | {r['steps']} | "
                     f"{r['prompt_tokens']}/{r['completion_tokens']} | {r['seconds']}s | {r['reason']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

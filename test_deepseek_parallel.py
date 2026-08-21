#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek 死活题并发测试：80 题分成 N 批，同步并发开跑。

用法：
  python test_deepseek_parallel.py                    # 默认：思考模式，10 批，max_tokens=131072
  python test_deepseek_parallel.py --batches 8        # 8 批（每批 10 题）
  python test_deepseek_parallel.py --batches 5        # 5 批（每批 16 题）
  python test_deepseek_parallel.py --no-think         # 关闭思考模式

环境变量：DEEPSEEK_API_KEY
"""

import json
import os
import sys
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    os.system('')  # 启用 Windows 控制台的 ANSI 转义序列
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

from test_deepseek import build_user_prompt, parse_answer, format_eta, extract_usage

BASE = Path(__file__).parent
JSON_PATH = BASE / "死活题.json"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(exist_ok=True)

BAR_W = 12  # 每批进度条宽度（字符数）


def main():
    parser = argparse.ArgumentParser(description="DeepSeek 死活题并发测试")
    parser.add_argument("--think", action="store_true", default=True, help="思考模式（默认开启）")
    parser.add_argument("--no-think", dest="think", action="store_false", help="关闭思考模式")
    parser.add_argument("--batches", type=int, default=10, help="并发批数（默认 10）")
    parser.add_argument("--ids", type=str, default=None, help="只测指定题号，逗号分隔，如 1,2,3,4,5")
    parser.add_argument("--max-tokens", type=int, default=131072, help="输出 token 上限")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DEEPSEEK_API_KEY")
        sys.exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        all_problems = json.load(f)["problems"]
    if args.ids:
        ids = {int(x.strip()) for x in args.ids.split(",")}
        all_problems = [p for p in all_problems if p["id"] in ids]
    total = len(all_problems)

    # ── 划分批次（顺序切分，保证每批内题号连续）──
    batch_size = (total + args.batches - 1) // args.batches
    batches = []
    for b in range(args.batches):
        seg = all_problems[b * batch_size:(b + 1) * batch_size]
        if seg:
            batches.append({
                "idx": b,
                "probs": seg,
                "id_range": (seg[0]["id"], seg[-1]["id"]),
                "size": len(seg),
            })
    n_batches = len(batches)  # 实际批数（total 不整除时可能少一批）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 全局状态（worker 更新 / renderer 读取，均加锁）──
    lock = threading.Lock()
    state = {
        "batches": [{
            "idx": b["idx"],
            "id_range": b["id_range"],
            "size": b["size"],
            "done": 0,
            "correct": 0,
            "empty": 0,
            "wrong": 0,
            "error": 0,
            "status": "pending",  # pending / running / done
        } for b in batches],
        "total_elapsed": 0.0,  # 已完成题的单题耗时之和，用于估算剩余时间
    }

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    t0 = time.time()

    # ── 单题处理 ──
    def run_one(prob):
        correct = prob["correctAnswer"]
        user_prompt = build_user_prompt(prob)
        kwargs = {"model": args.model,
                  "messages": [{"role": "user", "content": user_prompt}]}
        if args.think:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"},
                                    "reasoning_effort": "high"}
            if args.max_tokens:
                kwargs["max_tokens"] = args.max_tokens
        else:
            kwargs["temperature"] = 0.0
            kwargs["max_tokens"] = 128

        t_req = time.time()
        reasoning = ""
        raw_answer = ""
        error_msg = ""
        finish_reason = ""
        usage = None
        try:
            resp = client.chat.completions.create(**kwargs)
            reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
            raw_answer = resp.choices[0].message.content or ""
            finish_reason = getattr(resp.choices[0], "finish_reason", None) or ""
            usage = extract_usage(resp)
        except Exception as e:
            error_msg = str(e)
        elapsed = time.time() - t_req

        parsed = parse_answer(raw_answer) if raw_answer else None
        is_correct = (parsed == correct) if parsed else False

        return {
            "id": prob["id"],
            "correctAnswer": correct,
            "modelAnswer": parsed,
            "isCorrect": is_correct,
            "prompt": user_prompt,
            "reasoning": reasoning,
            "response": raw_answer,
            "error": error_msg or None,
            "finish_reason": finish_reason,
            "usage": usage,
            "elapsed": round(elapsed, 2),
        }

    # ── 批次处理：写独立文件，逐题更新状态 ──
    def run_batch(b):
        idx = b["idx"]
        out_path = RESULT_DIR / f"flash_batch{idx + 1:02d}_{ts}.jsonl"
        with lock:
            state["batches"][idx]["status"] = "running"
        with open(out_path, "w", encoding="utf-8") as jf:
            for prob in b["probs"]:
                rec = run_one(prob)
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jf.flush()
                with lock:
                    st = state["batches"][idx]
                    st["done"] += 1
                    state["total_elapsed"] += rec["elapsed"]
                    if rec["error"]:
                        st["error"] += 1
                    elif not rec["response"]:
                        st["empty"] += 1
                    elif rec["isCorrect"]:
                        st["correct"] += 1
                    else:
                        st["wrong"] += 1
        with lock:
            state["batches"][idx]["status"] = "done"

    # ── 状态面板 ──
    def build_panel():
        with lock:
            snap = [dict(b) for b in state["batches"]]
            total_elapsed = state["total_elapsed"]
        done = sum(b["done"] for b in snap)
        correct = sum(b["correct"] for b in snap)
        empty = sum(b["empty"] for b in snap)
        wrong = sum(b["wrong"] for b in snap)
        error = sum(b["error"] for b in snap)
        elapsed = time.time() - t0

        lines = [f"{args.model} {'思考' if args.think else '直接'}模式 · "
                 f"{n_batches} 批并发 · 共 {total} 题"]
        if done > 0:
            # 剩余时间 = 平均单题耗时 × 最慢批次还差的题数（并发下按最慢批估算）
            avg_single = total_elapsed / done
            max_remaining = max(b["size"] - b["done"] for b in snap)
            eta = avg_single * max_remaining
            eta_str = f"预计剩余 ~{format_eta(eta)}"
        else:
            eta_str = "预计剩余 ~估算中"
        lines.append(f"总进度 {done}/{total} ({done / total * 100:.1f}%) | "
                     f"对{correct} 空{empty} 错{wrong} 误{error} | "
                     f"已用 {format_eta(elapsed)} | {eta_str}")
        for b in snap:
            idx = b["idx"]
            lo, hi = b["id_range"]
            filled = int(BAR_W * b["done"] / b["size"]) if b["size"] else 0
            bar = "█" * filled + "░" * (BAR_W - filled)
            status = {"pending": "等待", "running": "运行", "done": "完成"}[b["status"]]
            lines.append(f"批{idx + 1:02d} 题{lo:02d}-{hi:02d} [{bar}] "
                         f"{b['done']}/{b['size']} {status} "
                         f"对{b['correct']}空{b['empty']}错{b['wrong']}误{b['error']}")
        return lines

    def write_panel(lines, first):
        if first:
            sys.stdout.write("\033[2J\033[H")  # 首次：清屏 + 回左上角
        else:
            sys.stdout.write("\033[H")         # 后续：回左上角重绘
        for line in lines:
            sys.stdout.write(line + "\033[K\n")  # \033[K 清到行尾，覆盖旧内容
        sys.stdout.flush()

    def renderer(stop_event):
        first = True
        while not stop_event.is_set():
            write_panel(build_panel(), first)
            first = False
            stop_event.wait(2.0)
        write_panel(build_panel(), first)  # 停止前最后渲染一次

    stop_event = threading.Event()
    render_thread = threading.Thread(target=renderer, args=(stop_event,), daemon=True)
    render_thread.start()

    # ── 并发跑各批（daemon 线程，Ctrl+C 可快速退出）──
    threads = []
    for b in batches:
        t = threading.Thread(target=run_batch, args=(b,), daemon=True)
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop_event.set()
        render_thread.join()
        print("\n\n已中断。已完成的部分已写入 results/ 下各批次文件。")
        print("未完成的题号可用 test_deepseek.py --id 补跑。")
        sys.exit(1)

    stop_event.set()
    render_thread.join()

    # ── 最终汇总 ──
    with lock:
        snap = [dict(b) for b in state["batches"]]
    done = sum(b["done"] for b in snap)
    correct = sum(b["correct"] for b in snap)
    empty = sum(b["empty"] for b in snap)
    wrong = sum(b["wrong"] for b in snap)
    error = sum(b["error"] for b in snap)
    acc = correct / done * 100 if done else 0.0

    print(f"\n{'=' * 60}")
    print("测试完成")
    print(f"  完成 {done}/{total} | 正确 {correct} ({acc:.1f}%) | "
          f"空 {empty} | 错 {wrong} | 调用失败 {error}")
    print(f"  总耗时 {format_eta(time.time() - t0)}")
    print(f"  结果: results/flash_batch01_{ts}.jsonl ~ flash_batch{n_batches:02d}_{ts}.jsonl")

    # ── 合并所有批次为一个总文件（按批顺序 → 题号顺序）──
    all_path = RESULT_DIR / f"flash_all_{ts}.jsonl"
    with open(all_path, "w", encoding="utf-8") as out:
        for idx in range(n_batches):
            bp = RESULT_DIR / f"flash_batch{idx + 1:02d}_{ts}.jsonl"
            if bp.exists():
                out.write(bp.read_text(encoding="utf-8"))
    print(f"  合并结果: {all_path.name}")


if __name__ == "__main__":
    main()

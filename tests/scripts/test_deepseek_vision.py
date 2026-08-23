#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek 视觉模型死活题测试（deepseek-v4-flash-vision-exp）

把每题棋盘图（boards/board_XXX.png）作为图片输入，连同题目要求与选项一起发给模型。
默认【不附带】黑白子坐标文本（模型需从图中读取局面）；--with-coords 可附带坐标以便对照。

用法：
  python test_deepseek_vision.py                      # 全部 80 题，思考模式，10 批并发
  python test_deepseek_vision.py --id 1               # 单题（打印 prompt 文本 + 模型回复）
  python test_deepseek_vision.py --id 1,5,23          # 指定题号
  python test_deepseek_vision.py --no-think           # 直接回答模式
  python test_deepseek_vision.py --batches 5          # 并发批数
  python test_deepseek_vision.py --with-coords        # 附带黑白子坐标文本
  python test_deepseek_vision.py --detail low         # 图片 detail: low/high/original/auto

环境变量：DEEPSEEK_API_KEY（或用 --api-key）
"""

import base64
import json
import os
import re
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

from test_deepseek import parse_answer, format_eta, extract_usage

BASE = Path(__file__).parent
JSON_PATH = BASE / "死活题.json"
BOARDS_DIR = BASE / "boards"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(exist_ok=True)

BAR_W = 12  # 每批进度条宽度（字符数）


def build_vision_messages(prob: dict, with_coords: bool, detail: str | None):
    """构造 OpenAI 兼容的 content 块数组（文本 + 图片）

    文本模板（与用户确认）：
        「如图所示，{题目要求（含坐标方向）}

        以下选项中，能成为正解的黑方第一手的坐标的是？

        选项：
          A：F2
          …」
    """
    header = prob["header"]
    black = prob["blackStones"]
    white = prob["whiteStones"]
    options = prob["options"]
    opts_text = "\n".join(f"  {opt}" for opt in options)

    # 从 header 拆出结尾问句（「以下选项中…」）
    marker = "以下选项中"
    idx = header.find(marker)
    if idx != -1:
        intro = header[:idx].rstrip()
        question = header[idx:].strip()
    else:
        intro = header.rstrip()
        question = ""

    parts = [f"如图所示，{intro.replace('一书中的题目，', '一书中的题目，单项选择。', 1)}"]
    if with_coords:
        parts.append(f"黑子：{black}\n白子：{white}")
    parts.append(f"{question}\n\n选项：\n{opts_text}\n\n"
                 "回复末尾请以 <box>X</box> 结尾，X 为你的选项字母（A~E）。")
    text = "\n\n".join(p for p in parts if p)

    # 读取棋盘图片 → base64 data URL
    img_path = BOARDS_DIR / f"board_{prob['id']:03d}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"缺少棋盘图片: {img_path}")
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    image_block = {"type": "image_url",
                   "image_url": {"url": f"data:image/png;base64,{b64}"}}
    if detail:
        image_block["image_url"]["detail"] = detail

    return [{"role": "user", "content": [{"type": "text", "text": text}, image_block]}], \
        text, str(img_path.name)


def extract_boxed_answer(text: str) -> str | None:
    """优先提取 <box>X</box> 中的答案字母；取最后一个匹配（指令要求放末尾）"""
    m = re.findall(r"<box>\s*([A-Ea-e])\s*</box>", text)
    if m:
        return m[-1].upper()
    return None


def main():
    parser = argparse.ArgumentParser(description="DeepSeek 视觉模型死活题测试")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash-vision-exp")
    parser.add_argument("--think", action="store_true", default=True, help="思考模式（默认开启）")
    parser.add_argument("--no-think", dest="think", action="store_false", help="关闭思考模式")
    parser.add_argument("--batches", type=int, default=10, help="并发批数（默认 10）")
    parser.add_argument("--ids", type=str, default=None, help="只测指定题号，逗号分隔")
    parser.add_argument("--with-coords", action="store_true", help="附带黑白子坐标文本")
    parser.add_argument("--detail", type=str, default=None,
                        help="图片 detail: low / high / original / auto")
    parser.add_argument("--max-tokens", type=int, default=131072, help="输出 token 上限")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DEEPSEEK_API_KEY 或用 --api-key")
        sys.exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        all_problems = json.load(f)["problems"]
    if args.ids:
        ids = {int(x.strip()) for x in args.ids.split(",")}
        all_problems = [p for p in all_problems if p["id"] in ids]
    total = len(all_problems)

    # 并发批：顺序切分，保证每批内题号连续
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
    n_batches = len(batches)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    single = len(batches) == 1 and total == 1  # 单题模式：打印 prompt 与回复

    lock = threading.Lock()
    state = {
        "batches": [{
            "idx": b["idx"], "id_range": b["id_range"], "size": b["size"],
            "done": 0, "correct": 0, "empty": 0, "wrong": 0, "error": 0,
            "status": "pending",
        } for b in batches],
        "total_elapsed": 0.0,
    }

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    t0 = time.time()

    def run_one(prob):
        correct = prob["correctAnswer"]
        try:
            messages, text_part, img_name = build_vision_messages(
                prob, args.with_coords, args.detail)
        except FileNotFoundError as e:
            return {"id": prob["id"], "correctAnswer": correct, "modelAnswer": None,
                    "isCorrect": False, "prompt": str(e), "response": "",
                    "error": str(e), "finish_reason": "", "usage": None,
                    "elapsed": 0.0, "image": "", "text_prompt": ""}

        kwargs = {"model": args.model, "messages": messages}
        if args.think:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"},
                                    "reasoning_effort": "high"}
            if args.max_tokens:
                kwargs["max_tokens"] = args.max_tokens
        else:
            kwargs["temperature"] = 0.0
            # 视觉模型即使不启用 thinking 也会先输出推理，128 token 会被耗尽；
            # 直接模式默认给足 4096，让推理+答案都能生成
            kwargs["max_tokens"] = args.max_tokens or 4096

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

        parsed = None
        if raw_answer:
            parsed = extract_boxed_answer(raw_answer) or parse_answer(raw_answer)
        is_correct = (parsed == correct) if parsed else False

        if single:
            print(f"\n{'=' * 60}")
            print(f"第 {prob['id']} 题 · 正确答案 {correct}")
            print(f"图片: {img_name}")
            print(f"{'─' * 60}")
            print("【Prompt 文本】")
            print(text_part)
            print(f"{'─' * 60}")
            if reasoning:
                print("【思考过程】")
                print(reasoning)
                print(f"{'─' * 60}")
            print("【模型回复】")
            print(raw_answer or f"(空) {error_msg}")
            print(f"{'=' * 60}\n")

        return {
            "id": prob["id"],
            "correctAnswer": correct,
            "modelAnswer": parsed,
            "isCorrect": is_correct,
            "prompt": text_part,
            "image": img_name,
            "reasoning": reasoning,
            "response": raw_answer,
            "error": error_msg or None,
            "finish_reason": finish_reason,
            "usage": usage,
            "elapsed": round(elapsed, 2),
        }

    def run_batch(b):
        idx = b["idx"]
        out_path = RESULT_DIR / f"vision_batch{idx + 1:02d}_{ts}.jsonl"
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
                 f"{n_batches} 批并发 · 共 {total} 题"
                 + (" · 附带坐标" if args.with_coords else "")]
        if done > 0:
            avg_single = total_elapsed / done
            max_remaining = max(b["size"] - b["done"] for b in snap)
            eta_str = f"预计剩余 ~{format_eta(avg_single * max_remaining)}"
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
            sys.stdout.write("\033[2J\033[H")
        else:
            sys.stdout.write("\033[H")
        for line in lines:
            sys.stdout.write(line + "\033[K\n")
        sys.stdout.flush()

    def renderer(stop_event):
        first = True
        while not stop_event.is_set():
            write_panel(build_panel(), first)
            first = False
            stop_event.wait(2.0)
        write_panel(build_panel(), first)

    stop_event = threading.Event()
    render_thread = threading.Thread(target=renderer, args=(stop_event,), daemon=True)
    render_thread.start()

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
        sys.exit(1)

    stop_event.set()
    render_thread.join()

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

    if n_batches > 1:
        all_path = RESULT_DIR / f"vision_all_{ts}.jsonl"
        with open(all_path, "w", encoding="utf-8") as out:
            for idx in range(n_batches):
                bp = RESULT_DIR / f"vision_batch{idx + 1:02d}_{ts}.jsonl"
                if bp.exists():
                    out.write(bp.read_text(encoding="utf-8"))
        print(f"  合并结果: {all_path.name}")


if __name__ == "__main__":
    main()

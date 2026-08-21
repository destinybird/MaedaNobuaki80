#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek 死活题测试脚本（支持思考模式）

用法：
  python test_deepseek.py                           # 普通模式，全部80题
  python test_deepseek.py --think                   # 思考模式 (reasoning)
  python test_deepseek.py --think --limit 10         # 思考模式，前10题
  python test_deepseek.py --think --start 40 --limit 40  # 第41~80题（分批）
  python test_deepseek.py --think --id 1,5,23        # 指定题号

环境变量：DEEPSEEK_API_KEY
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

BASE = Path(__file__).parent
JSON_PATH = BASE / "死活题.json"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(exist_ok=True)


def build_user_prompt(prob: dict) -> str:
    """构建用户 prompt — 把 header 里的问句「以下选项中…」后置到棋子坐标之后"""
    header = prob["header"]
    black = prob["blackStones"]
    white = prob["whiteStones"]
    options = prob["options"]
    opts_text = "\n".join(f"  {opt}" for opt in options)

    # 从 header 拆出结尾问句（「以下选项中，能成为正解的黑方第一手的坐标的是？」），后置
    marker = "以下选项中"
    idx = header.find(marker)
    if idx != -1:
        intro = header[:idx].rstrip()
        question = header[idx:].strip()
    else:
        intro = header.rstrip()
        question = ""

    return f"{intro}黑子：{black}\n白子：{white}\n\n{question}\n\n选项：\n{opts_text}"


def parse_answer(text: str) -> str | None:
    """从模型回复中提取答案字母 A-E"""
    text = text.strip()
    # 「答案：X」/「答案为X」/「答案是X」
    m = re.search(r'答案(?:是|为)?\s*[：:]?\s*\*{0,2}\s*([A-Ea-e])\b', text)
    if m:
        return m.group(1).upper()
    # 结论性表述：选X / 选择X / 正确答案X / 正确选项X / 正解X
    # 取最后一个匹配（结论通常写在文末）
    m = re.findall(r'(?:选择|正确答案|正确选项|正解|选)\s*(?:是|为)?\s*[：:]?\s*\*{0,2}\s*([A-Ea-e])\b', text)
    if m:
        return m[-1].upper()
    # 单独一行 A-E
    for line in reversed(text.split("\n")):
        line = line.strip()
        if re.match(r'^[A-Ea-e]$', line):
            return line.upper()
    # 最后一个孤立 A-E
    m = re.findall(r'\b([A-E])\b', text)
    if m:
        return m[-1]
    # 纯字母
    clean = text.replace(" ", "").replace("\n", "").replace(".", "")
    if len(clean) == 1 and clean.upper() in "ABCDE":
        return clean.upper()
    return None


def format_eta(seconds: float) -> str:
    # 先整体四舍五入成整数秒，再拆分时分秒。
    # 否则会出现 "1m60s" 这类错误（秒单独 :.0f 进位成 60）。
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def extract_usage(resp):
    """从响应中提取 token 用量（含 reasoning_tokens），失败返回 None"""
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    details = getattr(u, "completion_tokens_details", None)
    return {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
    }


def main():
    parser = argparse.ArgumentParser(description="DeepSeek 死活题测试")
    parser.add_argument("--think", action="store_true", help="启用思考模式 (reasoning)")
    parser.add_argument("--id", type=str, help="指定题号，逗号分隔")
    parser.add_argument("--start", type=int, default=0, help="跳过前 N 题（配合 --limit 分批）")
    parser.add_argument("--limit", type=int, help="最多测 N 题")
    parser.add_argument("--max-tokens", type=int, default=None, help="输出 token 上限（思考模式建议显式设大，如 131072）")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DEEPSEEK_API_KEY")
        sys.exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_problems = data["problems"]

    if args.id:
        ids = {int(x.strip()) for x in args.id.split(",")}
        problems = [p for p in all_problems if p["id"] in ids]
    else:
        problems = all_problems[args.start:]
        if args.limit:
            problems = problems[: args.limit]

    total = len(problems)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "think" if args.think else "direct"
    id_tag = ""
    if args.id:
        ids_sorted = sorted(int(x.strip()) for x in args.id.split(","))
        id_tag = "_id" + "-".join(str(i) for i in ids_sorted)
    out_path = RESULT_DIR / f"deepseek_{args.model}_{mode}{id_tag}_{ts}.jsonl"

    print(f"模型: {args.model}")
    print(f"模式: {'思考模式' if args.think else '直接回答'}")
    print(f"题目数: {total}")
    print(f"结果文件: {out_path.name}")
    print(f"{'=' * 60}")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    results = []
    correct_count = 0
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as jf:
        for i, prob in enumerate(problems):
            pid = prob["id"]
            n_done = i + 1
            correct = prob["correctAnswer"]
            user_prompt = build_user_prompt(prob)

            t_req_start = time.time()

            # 构建请求参数
            kwargs = dict(
                model=args.model,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if args.think:
                kwargs["extra_body"] = {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "high",
                }
                if args.max_tokens:
                    kwargs["max_tokens"] = args.max_tokens
            else:
                kwargs["temperature"] = 0.0
                kwargs["max_tokens"] = 128

            reasoning = ""
            raw_answer = ""
            error_msg = ""
            finish_reason = ""
            usage = None
            try:
                response = client.chat.completions.create(**kwargs)
                reasoning = getattr(response.choices[0].message, "reasoning_content", None) or ""
                raw_answer = response.choices[0].message.content or ""
                finish_reason = getattr(response.choices[0], "finish_reason", None) or ""
                usage = extract_usage(response)
            except Exception as e:
                error_msg = str(e)

            t_req = time.time() - t_req_start

            if raw_answer:
                parsed = parse_answer(raw_answer)
            else:
                parsed = None

            is_correct = (parsed == correct) if parsed else False
            if is_correct:
                correct_count += 1

            # 记录
            record = {
                "id": pid,
                "correctAnswer": correct,
                "modelAnswer": parsed,
                "isCorrect": is_correct,
                "prompt": user_prompt,
                "reasoning": reasoning,
                "response": raw_answer,
                "error": error_msg or None,
                "finish_reason": finish_reason,
                "usage": usage,
                "elapsed": round(t_req, 2),
            }

            # 写入文件
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            jf.flush()
            results.append(record)

            # 实时输出
            if not raw_answer:
                status = f"\033[91m✗ 空({finish_reason or '未知'})\033[0m"
            elif is_correct:
                status = "\033[92m✓ 正确\033[0m"
            else:
                status = f"\033[91m✗ 错误\033[0m (应为{correct}，答{parsed})"

            elapsed = time.time() - t0
            avg = elapsed / n_done
            eta = avg * (total - n_done)
            acc = correct_count / n_done * 100

            bar_w = 20
            filled = int(bar_w * n_done / total)
            bar = "█" * filled + "░" * (bar_w - filled)

            print(
                f"\r[{n_done:>3}/{total}] {bar} "
                f"第{pid:>2}题 {status}  "
                f"| \033[96m累计 {correct_count}/{n_done} {acc:.1f}%\033[0m"
                f"  | ⏱ 本{format_eta(t_req)} 均{format_eta(avg)} "
                f"剩余{format_eta(eta)}",
                end="",
                flush=True,
            )

            if n_done % 10 == 0 or n_done == total:
                print()

            if i < total - 1:
                time.sleep(0.5)

    elapsed = time.time() - t0
    accuracy = correct_count / total * 100 if total else 0

    print(f"\n{'=' * 60}")
    print(f"测试完成")
    print(f"  正确: {correct_count}/{total} ({accuracy:.1f}%)")
    print(f"  总耗时: {format_eta(elapsed)}")
    print(f"  结果: {out_path}")


if __name__ == "__main__":
    main()

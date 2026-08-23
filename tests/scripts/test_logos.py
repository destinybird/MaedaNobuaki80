#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoGos-7B（= InternThinker-Go）死活题测试脚本（vLLM 版）。

公平性：prompt 与 test_deepseek.py 的 build_user_prompt 完全一致；
max_tokens 上限 131072 与 DeepSeek 思考模式一致。
vLLM 的 stop=["</answer>"]：LoGos 输出"第一轮推理+<answer>X</answer>"后
不会停止（其 EOS 与 tokenizer 不匹配，会无限复读），stop 在第一个
</answer> 处立即截断，只砍复读、不影响答案，因此不构成对答案的干预。

用法：
  python test_logos.py --model ./LoGos-7B --id 1      # 单题
  python test_logos.py --model ./LoGos-7B             # 全量 80 题
  python test_logos.py --model ./LoGos-7B --start 0 --limit 20
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from test_deepseek import parse_answer, format_eta

BASE = Path(__file__).parent
JSON_PATH = BASE / "死活题.json"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(exist_ok=True)


def answer_with_coord(raw: str, prob: dict) -> str | None:
    """提取答案：优先解析 <box> 内内容（字母直接用、坐标映射选项）；
    再试 parse_answer；最后全文坐标映射"""
    # 1) <box> 内内容
    m = re.search(r"<box>(.*?)</box>", raw, re.S)
    if m:
        inner = m.group(1).strip()
        if re.fullmatch(r"[A-Ea-e]", inner):
            return inner.upper()
        cm = re.search(r"[A-Ta-t]\d{1,2}", inner)
        if cm:
            return _map_coord(cm.group(0).upper(), prob)
    # 2) parse_answer
    letter = parse_answer(raw)
    if letter:
        return letter
    # 3) 全文坐标
    m = re.search(r"[A-Ta-t]\d{1,2}", raw)
    if m:
        return _map_coord(m.group(0).upper(), prob)
    return None


def _map_coord(coord: str, prob: dict) -> str | None:
    """坐标命中 A~D 某选项坐标 → 返回该选项字母"""
    for opt in prob["options"]:
        parts = opt.replace("：", ":").split(":", 1)
        if len(parts) == 2 and parts[0].strip() != "E":
            if parts[1].strip().upper() == coord:
                return parts[0].strip().upper()
    return None


def build_prompt(prob: dict) -> str:
    """vision 版文本 prompt（对齐 vision 公平性）：题面（单项选择）+ 问句 + 选项 + <box> 指令"""
    header = prob["header"]
    options = prob["options"]
    opts_text = "\n".join(f"  {opt}" for opt in options)
    marker = "以下选项中"
    idx = header.find(marker)
    intro = header[:idx].rstrip() if idx != -1 else header.rstrip()
    question = header[idx:].strip() if idx != -1 else ""
    intro = intro.replace("一书中的题目，", "一书中的题目，单项选择。", 1)
    return f"{intro}\n\n{question}\n\n选项：\n{opts_text}\n\n" \
           "回复末尾请以 <box>X</box> 结尾，X 为你的选项字母（A~E）。"


def main():
    parser = argparse.ArgumentParser(description="LoGos-7B 死活题测试（vLLM）")
    parser.add_argument("--model", type=str, default="./LoGos-7B")
    parser.add_argument("--id", type=str, default=None, help="只测指定题号，逗号分隔")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="输出 token 上限。正常题 stop 在 <box> 块提前停；"
                             "复读题（不输出 <box>）2048 截断记空，避免无限复读")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="重复惩罚。实测惩罚会让 LoGos 直接输出垃圾（如 <box>和</box>），"
                             "默认 1.0 关闭；复读题靠 max_new_tokens 截断处理")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="vLLM 上下文窗口。LoGos config 声明 131072，A10 显存不够预留 KV；"
                             "实际序列很短（stop 提前截断），32768 足够")
    parser.add_argument("--gpu-mem", type=float, default=0.90, help="GPU 显存占用比例")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    print(f"加载模型: {args.model} (vLLM, max_model_len={args.max_model_len}) ...")
    t0 = time.time()
    llm = LLM(model=args.model, dtype="float16",
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len)
    print(f"加载完成 {format_eta(time.time()-t0)}")

    data = json.load(open(JSON_PATH, encoding="utf-8"))
    probs = data["problems"]
    if args.id:
        ids = {int(x) for x in args.id.split(",")}
        probs = [p for p in probs if p["id"] in ids]
    else:
        probs = probs[args.start:]
        if args.limit:
            probs = probs[: args.limit]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULT_DIR / f"logos_{ts}.jsonl"
    print(f"共 {len(probs)} 题 → {out.name}\n")

    # stop 在第一个 </box> 截断；repetition_penalty 打断复读循环
    sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0,
                        stop=["</box>"], include_stop_str_in_output=True,
                        repetition_penalty=args.repetition_penalty)

    n_correct = 0
    t_all = time.time()
    with open(out, "w", encoding="utf-8") as jf:
        for i, prob in enumerate(probs):
            pid = prob["id"]
            correct = prob["correctAnswer"]
            prompt = build_prompt(prob)  # vision 版文本 prompt（对齐 vision）

            t_start = time.time()
            t_str = time.strftime("%H:%M:%S")
            raw = ""
            try:
                outs = llm.generate([prompt], sp)
                raw = outs[0].outputs[0].text
            except Exception as e:
                print(f"  id={pid} 推理失败: {e}")
            elapsed = time.time() - t_start

            parsed = answer_with_coord(raw, prob) if raw else None
            is_correct = parsed == correct
            if is_correct:
                n_correct += 1

            rec = {
                "id": pid, "correctAnswer": correct, "modelAnswer": parsed,
                "isCorrect": is_correct, "prompt": prompt, "response": raw,
                "elapsed": round(elapsed, 1),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            status = "✓" if is_correct else ("空" if not parsed else "✗")
            done = i + 1
            eta = (time.time() - t_all) / done * (len(probs) - done)
            gen_tokens = len(outs[0].outputs[0].token_ids) if raw else 0
            print(f"[{t_str}] id={pid:>3} {status} 正解{correct} 答{parsed or '-'} "
                  f"耗时{elapsed:.0f}s 生成{gen_tokens}tok "
                  f"({gen_tokens/elapsed:.0f} tok/s) | "
                  f"累计{n_correct}/{done} 剩余~{format_eta(eta)}", flush=True)

    acc = n_correct / len(probs) * 100
    print(f"\n完成 {len(probs)} 题 | 正确 {n_correct} ({acc:.1f}%) | {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoGos-7B 死活题测试（vLLM，统一题面含棋盘坐标版）——2026-09 重测

背景：2026-08-23 首次测试的 prompt 是"与 Vision 版一致"的文字模板，但 LoGos 是
纯文本模型、没有图片通道，且题面里【没有】黑子/白子坐标行 → 模型根本看不到棋盘，
5.0% 的结果作废。本次重测使用与 DeepSeek-V4-Flash/Pro/Vision-Exp（2026-09-03）
完全相同的文字题面：单项选择题描述（单项选择）+ 黑子/白子坐标行 + 问句 +
A~E 选项 + `<box>X</box>` 输出指令。纯文本输入，不附图。

模型：LoGos-7B（= InternThinker-Go，NIPS 2025，基于 Qwen2.5-7B）
运行环境：ModelScope GPU Notebook（A10 24GB 实测），vLLM 本地推理。

与旧 test_logos.py 的关键差异：
1. prompt 增加"黑子/白子"两行（本次修复的核心）；
2. 答案只认回复末尾 <box> 内的字母（或坐标映射到选项）；无 <box> 一律记空，
   不做 parse_answer/坐标兜底——LoGos 复读会污染统计（8/23 教训：17.5% 虚高 →
   核查后 5.0%）。原始回复全文留档，供事后逐条人工核查。

部署要点（备忘已有，快速回顾）：
- 模型下载：export HF_ENDPOINT=https://hf-mirror.com（ModelScope 若无镜像）
- vLLM 显存：模型 config 声明 max_seq_len=131072，A10 不够预留 KV，必须显式
  max_model_len=32768
- 不在 <box> 处设 stop（不干预作答过程）。实测该模型没有 EOS、会一路复读停不下来，
  因此必须设生成上限：默认 max_new_tokens=2048（与 8 月口径一致，约 1 分钟/题），
  到上限即停（finish=length）——上限截断是模型自身停不下来所致，如实记录、事后
  人工分类。各题完整复读文本保留在 jsonl 里供核查。
- repetition_penalty 保持 1.0（关闭）：加惩罚会让 LoGos 输出 <box>和</box> 垃圾

用法：
  python test_logos_unified.py --id 1                          # 先单题验证
  python test_logos_unified.py                                 # 全量 80 题
  python test_logos_unified.py --start 0 --limit 40            # 分批（防实例1小时关闭）
  python test_logos_unified.py --start 40 --limit 40 --append  # 续跑（追加到同一文件，跳过已有id）
  python test_logos_unified.py --dry-run --id 1,9,79           # 只打印拼好的题面，不加载模型

输出：results_unified/logos_all_<时间戳>.jsonl（逐题 flush；--append 时续写到该文件）
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

BASE = Path(__file__).parent
JSON_PATH = BASE / "死活题.json"
RESULT_DIR = BASE / "results_unified"
RESULT_DIR.mkdir(exist_ok=True)

COLS = "ABCDEFGHJKLMNOPQRST"  # 19 列，不含 I


def build_text_prompt(prob: dict) -> str:
    """统一文字题面（与 test_deepseek_unified.py 逐字一致，勿改动）"""
    header = prob["header"]
    black = prob["blackStones"]
    white = prob["whiteStones"]
    options = prob["options"]
    opts_text = "\n".join(f"  {opt}" for opt in options)

    marker = "以下选项中"
    idx = header.find(marker)
    if idx != -1:
        intro = header[:idx].rstrip()
        question = header[idx:].strip()
    else:
        intro = header.rstrip()
        question = ""

    if "《痛快》一书中的题目，" in intro:
        intro = intro.replace("《痛快》一书中的题目，", "《痛快》一书中的题目，单项选择。", 1)
    elif "一书中的题目，" in intro:
        intro = intro.replace("一书中的题目，", "一书中的题目，单项选择。", 1)
    else:
        dot = intro.find("。")
        if dot != -1:
            intro = intro[:dot + 1] + "单项选择。" + intro[dot + 1:]
        else:
            intro = "单项选择。" + intro

    if question:
        return (f"{intro}\n黑子：{black}\n白子：{white}\n\n"
                f"{question}\n\n选项：\n{opts_text}\n\n"
                "回复末尾请以 <box>X</box> 结尾，X 为你的选项字母（A~E）。")
    return (f"{intro}\n黑子：{black}\n白子：{white}\n\n"
            f"选项：\n{opts_text}\n\n"
            "回复末尾请以 <box>X</box> 结尾，X 为你的选项字母（A~E）。")


def _map_coord(coord: str, prob: dict) -> str | None:
    """坐标命中 A~D 某选项坐标 → 返回该选项字母"""
    for opt in prob["options"]:
        parts = opt.replace("：", ":").split(":", 1)
        if len(parts) == 2 and parts[0].strip() != "E":
            if parts[1].strip().upper() == coord:
                return parts[0].strip().upper()
    return None


def extract_answer(text: str, prob: dict) -> str | None:
    """只认 <box> 内内容：字母直接取；坐标映射到选项。无 <box> 返回 None（记空）。
    不做正文兜底——LoGos 复读/正文枚举会污染统计（8/23 教训）。"""
    m = re.findall(r"<box>\s*([A-Ea-e])\s*</box>", text or "")
    if m:
        return m[-1].upper()
    m = re.findall(r"<box>\s*([A-Ta-t]\d{1,2})\s*</box>", text or "")
    if m:
        return _map_coord(m[-1].upper(), prob)
    return None


def format_eta(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(description="LoGos-7B 死活题测试（vLLM，统一题面含坐标版）")
    parser.add_argument("--model", type=str, default="./LoGos-7B",
                        help="模型路径/名称（ModelScope 或 HF 下载后本地路径）")
    parser.add_argument("--id", type=str, default=None, help="只测指定题号，逗号分隔")
    parser.add_argument("--start", type=int, default=0, help="跳过前 N 题（配合 --limit 分批）")
    parser.add_argument("--limit", type=int, default=None, help="最多测 N 题")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="生成 token 上限（该模型无 EOS 会一直复读，默认 2048 与 8 月"
                             "口径一致；到上限即停，约 1 分钟/题）")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="vLLM 上下文窗口（config 声明 131072 会 OOM，勿改大）")
    parser.add_argument("--gpu-mem", type=float, default=0.90, help="GPU 显存占用比例")
    parser.add_argument("--append", action="store_true",
                        help="续跑：追加到最近一次 logos_all_*.jsonl（跳过其中已有 id）")
    parser.add_argument("--dry-run", action="store_true", help="只打印题面，不加载模型")
    args = parser.parse_args()

    data = json.load(open(JSON_PATH, encoding="utf-8"))
    probs = data["problems"]

    if args.dry_run:
        if args.id:
            ids = {int(x) for x in args.id.split(",")}
            probs = [p for p in probs if p["id"] in ids]
        for p in probs:
            print("=" * 60)
            print("第 %d 题题面预览" % p["id"])
            print("-" * 60)
            print(build_text_prompt(p))
        print("=" * 60)
        print("dry-run 结束：以上为将发送给模型的完整文字题面。")
        return

    if args.id:
        ids = {int(x) for x in args.id.split(",")}
        probs = [p for p in probs if p["id"] in ids]
    else:
        probs = probs[args.start:]
        if args.limit:
            probs = probs[: args.limit]
    total = len(probs)
    if total == 0:
        print("没有要测的题")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.append:
        cands = sorted(RESULT_DIR.glob("logos_all_*.jsonl"))
        if not cands:
            print("--append 但没找到已有的 logos_all_*.jsonl")
            return
        out_path = cands[-1]
        done_ids = set()
        for ln in open(out_path, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                done_ids.add(json.loads(ln)["id"])
        probs = [p for p in probs if p["id"] not in done_ids]
        total = len(probs)
        print(f"续跑 {out_path.name}，已存在 {len(done_ids)} 条，本次再测 {total} 题")
    else:
        out_path = RESULT_DIR / f"logos_all_{ts}.jsonl"

    # 可选自检：若旁边有 test_deepseek_unified.py 则比对题面是否逐字一致；
    # 没有该文件时静默跳过（不打印，避免误导）
    if (BASE / "test_deepseek_unified.py").exists():
        try:
            import test_deepseek_unified as _dsu
            mismatch = [p["id"] for p in probs
                        if build_text_prompt(p) != _dsu.build_text_prompt(p)]
            if mismatch:
                print("警告：题面与 DeepSeek 统一版不一致的题号:", mismatch)
            else:
                print("自检通过：题面与 DeepSeek 统一版逐字一致")
        except Exception as e:
            print("（题面自检失败，可忽略）", e)

    from vllm import LLM, SamplingParams

    print(f"加载模型: {args.model} (vLLM, max_model_len={args.max_model_len}) ...")
    t0 = time.time()
    llm = LLM(model=args.model, dtype="float16",
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len)
    print("加载完成 %s" % format_eta(time.time() - t0))

    # 生成上限：该模型无 EOS、会复读到上限才停；默认 2048（约 1 分钟/题），
    # 不设 stop，不干预作答；截断只发生在模型自己停不下来时（finish=length）
    sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0,
                        repetition_penalty=1.0)

    n_correct = 0
    t_all = time.time()
    with open(out_path, "a" if args.append else "w", encoding="utf-8") as jf:
        for i, prob in enumerate(probs):
            pid = prob["id"]
            correct = prob["correctAnswer"]
            prompt = build_text_prompt(prob)

            t_start = time.time()
            raw = ""
            finish = ""
            n_tok = 0
            error_msg = ""
            try:
                outs = llm.generate([prompt], sp)
                o = outs[0].outputs[0]
                raw = o.text
                finish = o.finish_reason or ""
                n_tok = len(o.token_ids)
            except Exception as e:
                error_msg = str(e)
            elapsed = time.time() - t_start

            parsed = extract_answer(raw, prob) if raw else None
            is_correct = (parsed == correct) if parsed else False
            if is_correct:
                n_correct += 1

            rec = {
                "id": pid, "correctAnswer": correct, "modelAnswer": parsed,
                "isCorrect": is_correct, "prompt": prompt, "response": raw,
                "finish_reason": finish or ("error" if error_msg else ""),
                "error": error_msg or None,
                "elapsed": round(elapsed, 1),
                "gen_tokens": n_tok,
                "tok_per_s": round(n_tok / elapsed, 1) if elapsed > 0 else 0.0,
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            status = "✓" if is_correct else ("空" if not parsed else "✗")
            done = i + 1
            eta = (time.time() - t_all) / done * (total - done)
            print("[%s] id=%-3d %s 正解%s 答%s finish=%s 耗时%.0fs %dtok %.0ftok/s | "
                  "累计%d/%d 剩余~%s" % (
                      time.strftime("%H:%M:%S"), pid, status, correct,
                      parsed or "-", finish or "-", elapsed, n_tok,
                      rec["tok_per_s"], n_correct, done, format_eta(eta)),
                  flush=True)

    acc = n_correct / total * 100
    print("\n本次完成 %d 题 | 正确 %d (%.2f%%) | 总耗时 %s" % (
        total, n_correct, acc, format_eta(time.time() - t_all)))
    print("结果文件:", out_path)


if __name__ == "__main__":
    main()

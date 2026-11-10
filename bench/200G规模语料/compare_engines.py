#!/usr/bin/env python3
"""三方编码对照：bbpe / tiktoken / HuggingFace tokenizers。

同一份文本、同一进程数、同一分块策略（复用 bench_encode_big 的按字节切分与
"逐块丢弃 ids" worker，避免上亿 id 撑爆 Python 堆而误测内存分配）。

用法:
  python compare_engines.py --text heldout.txt \
      --bbpe-model output/vocab_150256.json \
      --hf-model output/hf_bpe_150256.json \
      --tiktoken cl100k_base --procs 8 --json output/compare_engines.json

  # 也可以直接从 jsonl 语料切一段文本（省掉手工准备）
  python compare_engines.py --from-jsonl data/heldout.jsonl --max-gb 1 ...

注意（公平性）：
- HF 用自己的默认配置（ByteLevel BPE + GPT-2 风格预分词），bbpe 用中文优化正则；
  这是"各自开箱默认"的比法，预分词差异会在结论里标注。
- 绝对吞吐会随机器状态漂移（实测同配置不同时刻可差 20%），所以**只在同一批次内
  比较比值**，报告里同时输出绝对值和相对 bbpe 的倍数。
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bench_encode_big as B  # noqa: E402  复用切分与 worker


def build_text_from_jsonl(src: str, max_gb: float, out: str) -> str:
    """从 jsonl 抽 text 字段拼成纯文本，用于编码对照（默认 1GB 足够稳定）。"""
    budget = max_gb * (1024 ** 3)
    total = 0
    with open(src, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
        for line in f:
            try:
                t = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue
            w.write(t)
            w.write("\n")
            total += len(t.encode()) + 1
            if total >= budget:
                break
    return out


def run_one(impl: str, model: str, text: str, procs: int, chunk_mb: int) -> tuple:
    jobs = [(impl, model, text, s, e, chunk_mb << 20)
            for s, e in B.cut_points(text, procs)]
    t0 = time.monotonic()
    if procs == 1:
        out = [B.work(j) for j in jobs]
    else:
        with mp.Pool(procs) as pool:
            out = pool.map(B.work, jobs)
    dt = time.monotonic() - t0
    nbytes = sum(x[0] for x in out)
    ntok = sum(x[1] for x in out)
    return nbytes, ntok, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="现成的纯文本文件")
    g.add_argument("--from-jsonl", help="从 jsonl 抽 text 字段现拼一份")
    ap.add_argument("--max-gb", type=float, default=1.0, help="--from-jsonl 时的文本上限")
    ap.add_argument("--text-out", default=None, help="--from-jsonl 时文本落盘位置")
    ap.add_argument("--bbpe-model", required=True)
    ap.add_argument("--hf-model", default=None, help="缺省则跳过 HF 一栏")
    ap.add_argument("--tiktoken", default="cl100k_base")
    ap.add_argument("--skip-tiktoken", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    text = a.text
    if a.from_jsonl:
        text = a.text_out or os.path.join(HERE, "output", "compare_text.txt")
        os.makedirs(os.path.dirname(text), exist_ok=True)
        build_text_from_jsonl(a.from_jsonl, a.max_gb, text)
    if not os.path.isfile(text):
        print(f"❌ 文本不存在: {text}", file=sys.stderr)
        return 1

    cases = [("bbpe", a.bbpe_model)]
    if not a.skip_tiktoken:
        cases.append(("tiktoken", a.tiktoken))
    if a.hf_model:
        if not os.path.isfile(a.hf_model):
            print(f"❌ HF 模型不存在: {a.hf_model}", file=sys.stderr)
            return 1
        cases.append(("hf", a.hf_model))

    results = []
    skipped = []
    print(f"\n文本 {os.path.getsize(text)/1e9:.3f} GB | 进程数 {a.procs}\n")
    for impl, model in cases:
        try:
            nbytes, ntok, dt = run_one(impl, model, text, a.procs, a.chunk_mb)
        except ImportError as e:
            # 缺依赖（tiktoken / tokenizers 未装）不应静默：记录并显式告知
            print(f"  ⚠️ 跳过 {impl}：{e}（安装：uv pip install --python .venv-bench/bin/python "
                  f"{'tiktoken' if impl == 'tiktoken' else 'tokenizers'}）")
            skipped.append({"impl": impl, "reason": str(e)})
            continue
        results.append({
            "impl": impl,
            "model": model,
            "seconds": round(dt, 2),
            "bytes": nbytes,
            "tokens": ntok,
            "mb_per_s": round(nbytes / dt / 1e6, 2),
            "mtok_per_s": round(ntok / dt / 1e6, 2),
            "bytes_per_token": round(nbytes / ntok, 3),
        })
        r = results[-1]
        print(f"  {impl:9s} {r['seconds']:7.1f}s | {r['mb_per_s']:6.2f} MB/s | "
              f"{r['mtok_per_s']:6.2f} M tok/s | {r['bytes_per_token']:5.3f} B/tok")

    if len(results) < 2:
        print("❌ 可用实现不足 2 个，无法形成对照", file=sys.stderr)
        return 1

    base = next((r for r in results if r["impl"] == "bbpe"), None)
    if base:
        for r in results:
            r["speedup_vs_bbpe"] = round(r["mb_per_s"] / base["mb_per_s"], 2) if base["mb_per_s"] else None
            r["token_saving_vs_bbpe"] = round(r["tokens"] / base["tokens"], 2) if base["tokens"] else None

        print("\n| 实现 | 吞吐 | 相对 bbpe | B/token | 同文本 token 数（相对 bbpe） |")
        print("|---|---|---|---|---|")
        for r in results:
            print(f"| {r['impl']} | {r['mb_per_s']} MB/s | {r['speedup_vs_bbpe']}× | "
                  f"{r['bytes_per_token']} | {r['tokens']:,} ({r['token_saving_vs_bbpe']}×) |")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({
                "text": text,
                "text_bytes": os.path.getsize(text),
                "procs": a.procs,
                "chunk_mb": a.chunk_mb,
                "results": results,
                "skipped": skipped,
                "note": "绝对吞吐随机器状态漂移，跨批次只比比值",
            }, f, ensure_ascii=False, indent=2)
        print(f"\n报告: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

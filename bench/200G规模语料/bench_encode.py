#!/usr/bin/env python3
"""bench_encode.py — 编解码性能基准（对应 README §5.3）。

测什么（每个维度单独记录，见 README §5.1 表）：
  1. 吞吐：encode_ordinary 单线程 MB/s；encode_ordinary_parallel 在
     1/4/8/16 线程下的 MB/s 与加速比
  2. 延迟：单行 encode 的 P50/P99（KB 级行，模拟在线推理）
  3. 内存：编码大文本的峰值 RSS（/proc/self/status VmHWM）
  4. 正确性：round-trip 无损（与吞吐测试同一份数据，失败即基准无效）

用法：
  python3 bench_encode.py --model output/vocab_100k.json \
      --input data/bench_text.txt --threads 1,4,8,16 --repeat 3 \
      --json output/bench_encode.json

数据说明：--input 给一份文本（行级），脚本会重复拼接直到 ≥ --min-bytes
（默认 50MB），保证吞吐测量不受小样本噪声影响。

R4-10：标准安装下导入路径是 `bbpe_tokenizer.bbpe_rs`（pyproject.toml 的
module-name），原脚本只试顶层 `bbpe_rs`，失败即整体跳过性能步骤。
现按顺序尝试三个路径，并可用 `--module` 显式指定。

R4-25：吞吐基准文本必须由**解析后的 text 字段**拼接 —— 原实现直接读原始
文件行，jsonl 输入会把 `{"text": ...}` 的 JSON 语法字符（引号/花括号/键名）
一起编码进基准，测的是"JSON 语料"而不是真实文本。

R5-4：jsonl 坏行 / 缺 text 字段不再直接抛异常崩溃，而是**跳过并计数**
（与 run_pipeline_hf.py 的 m17 范式一致）；全部行无效时才以 1 退出。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

# R4-10：按优先级尝试的导入路径（标准安装在包内；--module 可覆盖）
IMPORT_CANDIDATES = ("bbpe_tokenizer.bbpe_rs", "bbpe_rs")


def load_bindings(module: str | None):
    """返回 Rust 绑定模块；全部失败返回 None。"""
    for name in ([module] if module else IMPORT_CANDIDATES):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


bbpe_rs = load_bindings(None)


def peak_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def parse_jsonl_texts(raw_lines) -> tuple[list[str], dict[str, int]]:
    """R5-4：解析 jsonl 行取 text 字段，坏行/缺字段**跳过并计数**。

    原实现 `[json.loads(l)["text"] for l in f if l.strip()]` 遇到一行坏 JSON
    或缺 text 字段就整体崩溃 —— 200G 语料里几行脏数据不应让基准跑不起来。
    与 run_pipeline_hf.py 的 m17 范式一致：跳过 + 计数 + 结束后讲清楚。
    抽为纯函数（消费可迭代对象）便于单测。

    返回 (有效文本行, 计数 dict)；计数含 ok / bad_json / no_text / empty。
    """
    lines: list[str] = []
    counts = {"ok": 0, "bad_json": 0, "no_text": 0, "empty": 0}
    for line in raw_lines:
        if not line.strip():
            counts["empty"] += 1
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            counts["bad_json"] += 1
            continue
        text = obj.get("text") if isinstance(obj, dict) else None
        if not isinstance(text, str):
            counts["no_text"] += 1
            continue
        lines.append(text)
        counts["ok"] += 1
    return lines, counts


def build_bench_text(lines: list[str], min_bytes: int, cap_bytes: int) -> str:
    """R4-25：由**已解析的文本行**拼接基准文本。

    原实现自己重新读原始文件：jsonl 输入时把整行 JSON（含引号/花括号/键名）
    当作文本编码，吞吐数字测的是"JSON 语料"而非真实文本。
    现在统一消费 main() 里解析好的 lines。
    """
    if not lines:
        raise SystemExit("❌ 输入为空")
    out = []
    total = 0
    i = 0
    while total < min_bytes:
        chunk = lines[i % len(lines)]
        out.append(chunk)
        total += len(chunk.encode("utf-8"))
        i += 1
        if total >= cap_bytes:
            break
    return "\n".join(out)


def bench_throughput(tok, text: str, threads: int, repeat: int):
    best = None
    n_tokens = 0
    for _ in range(repeat):
        t0 = time.perf_counter()
        if threads == 1:
            ids = tok.encode_ordinary(text)
        else:
            ids = tok.encode_ordinary_parallel(text, num_threads=threads)
        dt = time.perf_counter() - t0
        mbps = len(text.encode("utf-8")) / 1e6 / dt
        best = mbps if best is None else max(best, mbps)
        n_tokens = len(ids)
    return best, n_tokens


def bench_latency(tok, lines, repeat: int):
    """单行延迟分布（在线推理场景）。"""
    samples = []
    for _ in range(repeat):
        for line in lines:
            t0 = time.perf_counter()
            tok.encode_ordinary(line)
            samples.append((time.perf_counter() - t0) * 1e3)  # ms
    samples.sort()
    n = len(samples)
    return {
        "n": n,
        "p50_ms": round(samples[n // 2], 4),
        "p99_ms": round(samples[min(n - 1, int(n * 0.99))], 4),
        "max_ms": round(samples[-1], 4),
        "line_bytes_avg": round(sum(len(l.encode("utf-8")) for l in lines) / len(lines), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="编解码性能基准")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="行级文本（jsonl:text 或 text）")
    ap.add_argument("--format", choices=["jsonl", "text"], default="text")
    ap.add_argument("--threads", default="1,4,8,16")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--min-bytes", type=int, default=50 * 1024 * 1024)
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--module",
        default=None,
        help="Rust 绑定的导入路径（默认依次尝试 bbpe_tokenizer.bbpe_rs / bbpe_rs）",
    )
    args = ap.parse_args()

    bbpe_rs = load_bindings(args.module)
    if bbpe_rs is None:
        tried = args.module or " / ".join(IMPORT_CANDIDATES)
        print(f"❌ 导入 Rust 绑定失败（已尝试: {tried}）：先 maturin develop --release",
              file=sys.stderr)
        return 2
    tok = bbpe_rs.BBPETokenizer(args.model)

    if args.format == "jsonl":
        with open(args.input, encoding="utf-8") as f:
            lines, counts = parse_jsonl_texts(f)
        bad = counts["bad_json"] + counts["no_text"]
        if bad:
            print(
                f"  ⚠️ 跳过无效行: 坏 JSON {counts['bad_json']} | 缺 text {counts['no_text']}"
                f"（有效 {counts['ok']} 行，空行 {counts['empty']}）",
                file=sys.stderr,
            )
    else:
        with open(args.input, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
    if not lines:
        print("❌ 输入为空", file=sys.stderr)
        return 1

    # R4-25：基准文本由解析后的 lines 拼接（不再重新读原始文件）
    text = build_bench_text(lines, args.min_bytes, args.min_bytes * 2)
    n_bytes = len(text.encode("utf-8"))
    print(f"== bench_encode ==\n  模型: {args.model}")
    print(f"  基准文本: {n_bytes / 1e6:.1f} MB（来自 {len(lines)} 行拼接）")

    report = {"model": args.model, "bench_bytes": n_bytes, "threads": {}}
    base = None
    rss0 = peak_rss_mb()

    for th in [int(x) for x in args.threads.split(",")]:
        mbps, n_tok = bench_throughput(tok, text, th, args.repeat)
        if th == 1:
            base = mbps
        speedup = mbps / base if base else float("nan")
        report["threads"][th] = {
            "mb_per_s": round(mbps, 2),
            "tokens": n_tok,
            "speedup_vs_1t": round(speedup, 2),
        }
        print(f"  线程 {th:>2}: {mbps:8.2f} MB/s | {n_tok:>10} tokens | 加速比 {speedup:.2f}x")

    lat = bench_latency(tok, lines[:500], max(1, args.repeat // 2 + 1))
    report["latency"] = lat
    print(f"  延迟: P50 {lat['p50_ms']}ms | P99 {lat['p99_ms']}ms | 平均行长 {lat['line_bytes_avg']}B")

    # 正确性：round-trip（基准有效性的前提）
    ids = tok.encode_ordinary(text)
    back = tok.decode(ids)
    report["roundtrip_ok"] = back == text
    print(f"  round-trip: {'✅ 无损' if back == text else '❌ 失败'}")
    if not report["roundtrip_ok"]:
        print("  ⚠️ round-trip 失败，吞吐数据不可信", file=sys.stderr)

    rss1 = peak_rss_mb()
    report["peak_rss_mb"] = round(max(rss0, rss1), 1)
    print(f"  峰值 RSS: {report['peak_rss_mb']} MB")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  报告: {args.json}")
    return 0 if report["roundtrip_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

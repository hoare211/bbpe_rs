#!/usr/bin/env python3
"""eval_quality.py — 分词器正确性 + 质量评估（对应 README §4.3/§4.4）。

评估项：
  1. round-trip 无损：decode(encode(t)) == t（覆盖 emoji / 生僻字 / 中英混排 / 代码片段）
  2. offsets 契约：单调、不重叠、text[s:e] 拼接 == 原文
  3. special token 行为：未授权被打碎；--allowed-special 白名单命中整体成词
  4. 质量指标：fertility（bytes/token）、字节级 token 占比、token 长度分布
     （字节级占比高 = 长尾碎裂，说明语料分布或 K 选型有问题）

用法：
  python3 eval_quality.py --model output/vocab_100k.json \
      --input data/eval_set.jsonl --format jsonl \
      --json output/quality_report.json

依赖：`pip install bbpe_rs` 的项目内 Python 绑定（maturin develop --release 后
`import bbpe_rs` 可用；装在 bbpe_tokenizer 包内时按实际路径导入）。

R4-10：标准安装（pyproject.toml 声明 `module-name = "bbpe_tokenizer.bbpe_rs"`）
下顶层 `import bbpe_rs` **不存在**，实测导入路径是 `bbpe_tokenizer.bbpe_rs`。
原脚本只试顶层路径，失败即整体跳过，质量步骤被静默略过。
现在按顺序尝试三个路径，并可用 `--module` 显式指定。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
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

# 覆盖“罕见但存在”的样本形态：推理遇到这些若被打碎成字节就是灾难
HARD_CASES = [
    "你好，世界！这是一个测试。",
    "中英混排 mixed text with English words 和数字 12345。",
    "生僻字：龘鱀爩癵驫讟麤纞；繁体：資訊科學。",
    "emoji：😀🚀🎉👨‍👩‍👧‍👦🇨🇳；组合：éèê ß ü ö",
    "数学：∑_{i=1}^{n} x² + √(a²+b²) ≠ ∫ f(x)dx",
    "代码：fn main() { println!(\"hello\"); } // TODO: fix",
    "跨行\n换行\t制表符   多空格   \n\n  连续空行",
    " 前后空白保留 ",
]


def read_lines(path: str, fmt: str, limit_bytes: int | None):
    total = 0
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if fmt == "jsonl":
                try:
                    obj = json.loads(line)
                    text = obj.get("text")
                    if not isinstance(text, str):
                        continue
                except json.JSONDecodeError:
                    continue
            else:
                text = line
            yield text
            total += len(text.encode("utf-8"))
            if limit_bytes and total >= limit_bytes:
                return


def check_roundtrip(tok, texts):
    bad = 0
    for t in texts:
        ids = tok.encode_ordinary(t)
        back = tok.decode(ids)
        if back != t:
            bad += 1
            if bad <= 3:
                print(f"  ❌ round-trip 失败: {t[:60]!r} → {back[:60]!r}", file=sys.stderr)
    return bad


def rebuild_from_offsets(text: str, offsets) -> str:
    """R4-13 / R5-5：按**字节偏移**从原文重建文本（纯函数，供单测）。

    offsets 是 Rust 侧语义的字节区间 `[start, end)`，必须按 `t.encode("utf-8")`
    切片；原实现 `t[s:e]` 按 Python 字符下标切，非 ASCII 文本必然错位。

    **先拼接全部字节再整体解码**（而非逐区间解码）：字节级 BPE 的 merge
    token 可以跨越字符边界（如小词表下 [E4 B8] 这样的 2 字节 token 是“你”
    的前两字节），逐区间 decode 会对合法编码报 UnicodeDecodeError ⟹ 误报
    offsets 不一致。越界 / 落在字符内部的**整体**非法切片仍返回 ""，
    由调用方判为不一致；逐区间的边界合法性由 check_offsets 另行检查。
    """
    tb = text.encode("utf-8")
    try:
        return b"".join(tb[s:e] for s, e in offsets).decode("utf-8")
    except UnicodeDecodeError:
        return ""


def check_offsets(tok, texts):
    bad = 0
    for t in texts:
        enc = tok.encode(t)
        # R4-13：offsets 是**字节偏移**（Rust 侧语义），必须按字节切片再解码。
        # 重建逻辑抽为 rebuild_from_offsets 纯函数（R5-5 可单测）。
        tb = t.encode("utf-8")
        rebuilt = rebuild_from_offsets(t, enc.offsets)
        if rebuilt != t:
            bad += 1
            if bad <= 3:
                print(f"  ❌ offsets 不一致: {t[:60]!r}", file=sys.stderr)
            continue
        prev_end = -1
        for s, e in enc.offsets:
            if not (0 <= s < e <= len(tb)) or s < prev_end:
                bad += 1
                break
            prev_end = e
    return bad


def load_special_tokens(model_path: str) -> list[str]:
    """从模型 JSON 读 added_tokens（Python 绑定的 tokenizer 未暴露该字段）。"""
    try:
        with open(model_path, encoding="utf-8") as f:
            data = json.load(f)
        return list((data.get("added_tokens") or {}).keys())
    except (OSError, json.JSONDecodeError):
        return []


def check_special(tok, specials):
    """special 语义：未授权被打碎为字节；授权后整体成词。

    R4-13：返回值含 `failed` 布尔量，供 main() 计入最终 ok —— 原实现的
    special 检查结果**不进入**退出判据，special 异常时评估仍以 0 退出。
    """
    if not specials:
        return {"skipped": "模型无 added_tokens", "failed": False}
    s0 = specials[0]
    probe = f"前文{s0}后文"
    # 未授权（encode_ordinary 拒绝一切 special）：special 文本不应作为整体 token 出现
    toks_plain = tok.encode(probe).tokens
    results = {
        "probe_special": s0,
        "unauthorized_fragmented": s0 not in toks_plain,
    }
    # 授权：白名单命中 ⟹ 整体成词
    try:
        enc_allowed = tok.encode(probe, allowed_special=[s0])
        results["authorized_atomic"] = s0 in enc_allowed.tokens
    except Exception as e:  # noqa: BLE001
        results["authorized_atomic"] = f"error: {e}"
    # 任一语义不满足 ⟹ special 门禁失败
    results["failed"] = not (
        results["unauthorized_fragmented"] is True
        and results["authorized_atomic"] is True
    )
    return results


def quality_metrics(tok, texts):
    total_bytes = 0
    total_tokens = 0
    byte_level = 0
    len_hist = Counter()
    for t in texts:
        enc = tok.encode(t)
        # 必须缓存：enc.ids / enc.offsets 每次访问都会把 Rust 侧向量重新物化成
        # Python list（O(n)）。在循环里写 enc.offsets[i] 会让单行退化成 O(n²)——
        # 实测 700 token 的行 0.027s、5 万 token 的行 ~137s，整份 50MB 评估集
        # 因此从几十秒涨到 40 分钟以上。
        ids = enc.ids
        offs = enc.offsets
        b = len(t.encode("utf-8"))
        total_bytes += b
        total_tokens += len(ids)
        for i in range(len(ids)):
            tb = offs[i][1] - offs[i][0]
            len_hist[min(tb, 16)] += 1
            if tb == 1:
                byte_level += 1
    n = max(total_tokens, 1)
    return {
        "bytes_per_token": round(total_bytes / n, 4),
        "byte_level_token_ratio": round(byte_level / n, 4),
        "token_len_hist_≤16": dict(sorted(len_hist.items())),
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="分词器正确性 + 质量评估")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="评估集（jsonl:text 或 text）")
    ap.add_argument("--format", choices=["jsonl", "text"], default="jsonl")
    ap.add_argument("--limit-bytes", type=int, default=50 * 1024 * 1024)
    ap.add_argument("--json", default=None, help="报告输出路径")
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

    texts = list(read_lines(args.input, args.format, args.limit_bytes))
    if not texts:
        print("❌ 评估集为空", file=sys.stderr)
        return 1
    print(f"== eval_quality ==\n  模型: {args.model}\n  评估行数: {len(texts)}")

    report = {"model": args.model, "input": args.input, "n_lines": len(texts)}

    rt_bad = check_roundtrip(tok, HARD_CASES + texts[:2000])
    off_bad = check_offsets(tok, texts[:2000])
    report["roundtrip"] = {"checked": len(HARD_CASES) + min(len(texts), 2000), "failed": rt_bad}
    report["offsets"] = {"checked": min(len(texts), 2000), "failed": off_bad}
    report["special"] = check_special(tok, load_special_tokens(args.model))
    report["quality"] = quality_metrics(tok, texts)

    # R4-13：special 门禁结果必须计入最终 ok（原实现不计入 ⟹ special 异常
    # 仍以 0 退出，CI 把失败判为通过）
    ok = rt_bad == 0 and off_bad == 0 and not report["special"].get("failed", False)
    print(f"  round-trip: {report['roundtrip']['checked'] - rt_bad}/{report['roundtrip']['checked']} 通过")
    print(f"  offsets:    {report['offsets']['checked'] - off_bad}/{report['offsets']['checked']} 通过")
    print(f"  special:    {report['special']}")
    print(f"  bytes/token: {report['quality']['bytes_per_token']} | "
          f"字节级 token 占比: {report['quality']['byte_level_token_ratio']:.2%}")
    print("  ✅ 正确性通过" if ok else "  ❌ 正确性未通过")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

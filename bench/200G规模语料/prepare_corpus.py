#!/usr/bin/env python3
"""prepare_corpus.py — 200G 语料训练前的盘点与格式统一。

作用（对应 README §3.2）：
  1. 把多种来源的原始语料（.txt / .jsonl / .json 数组）统一转换为
     `bbpe train` 认识的输入：jsonl（每行 {"text": "..."}）或纯文本（逐行）；
  2. 做**进训练前必须做**的预处理：UTF-8 严格校验（剔除非法字节行）、
     空行/过短行过滤、超长行截断（避免预分词 chunk 被强切的意外）；
  3. 产出 manifest：行数、字节数、字符集覆盖（CJK/拉丁/数字/emoji）、
     跳过原因统计 —— 后续一切评估以这份 manifest 为基线。

刻意**不做**的事（README §2.4 B）：不做繁简/全半角/大小写规范化，
不剔除特殊符号。推理侧不会做同样处理，规范化只会让 fertility 虚高。

用法：
  python3 prepare_corpus.py \
      --input /data/raw/ --output /data/corpus/train.jsonl \
      --format jsonl --report bench/200G规模语料/output/corpus_manifest.json

  # 混合目录（txt 按行、json 按数组递归扫描）：
  python3 prepare_corpus.py --input /data/raw --output /data/corpus/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# 字符大类判定（粗粒度，只用于 manifest 盘点，不修改文本）
EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0xFE00, 0xFE0F),
)


def char_class(ch: str) -> str:
    cp = ord(ch)
    for lo, hi in EMOJI_RANGES:
        if lo <= cp <= hi:
            return "emoji"
    if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿":
        return "cjk"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    return "other"


def iter_source_lines(path: Path):
    """把一个原始文件逐行产出文本。

    .jsonl  : 每行 JSON，取 text/conversations 字段（与 reader 语义对齐）
    .json   : 顶层数组，逐元素取 text
    .txt/其他: 逐物理行
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        yield ("__BAD_JSON__", line)
                        continue
                    text = obj.get("text") if isinstance(obj, dict) else None
                    if not isinstance(text, str):
                        if isinstance(obj, dict) and isinstance(obj.get("conversations"), list):
                            parts = [
                                c.get("content", "")
                                for c in obj["conversations"]
                                if isinstance(c, dict) and isinstance(c.get("content"), str)
                            ]
                            text = "\n".join(p for p in parts if p)
                    yield ("__OK__", text if isinstance(text, str) else None)
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for obj in items:
                if isinstance(obj, dict):
                    text = obj.get("text")
                    if isinstance(text, str):
                        yield ("__OK__", text)
                    else:
                        yield ("__BAD_JSON__", None)
                elif isinstance(obj, str):
                    yield ("__OK__", obj)
        else:
            with path.open("r", encoding="utf-8", errors="strict") as f:
                for line in f:
                    yield ("__OK__", line.rstrip("\r\n"))
    except (UnicodeDecodeError, OSError) as e:
        # 整个文件不可读（非法 UTF-8 / IO 错误）：记录后跳过，不让流水线中断
        yield ("__BAD_FILE__", f"{path}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="200G 语料盘点 + 格式统一（jsonl/text）")
    ap.add_argument("--input", required=True, help="原始语料文件或目录（递归扫描）")
    ap.add_argument("--output", required=True, help="输出语料路径（jsonl 或 text）")
    ap.add_argument("--format", choices=["jsonl", "text"], default="jsonl")
    ap.add_argument("--report", default=None, help="manifest 输出路径（json）")
    ap.add_argument("--min-chars", type=int, default=2, help="短于该长度的行剔除")
    ap.add_argument("--max-chars", type=int, default=100_000, help="超长行截断")
    ap.add_argument("--limit-bytes", type=int, default=None, help="最多输出字节数（调试用）")
    args = ap.parse_args()

    src = Path(args.input)
    files = (
        [p for p in src.rglob("*") if p.is_file()]
        if src.is_dir()
        else [src]
    )
    files.sort()
    if not files:
        print(f"❌ 输入为空：{src}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    char_classes = Counter()
    total_bytes = 0
    written = 0
    per_source = {}

    with out_path.open("w", encoding="utf-8") as out:
        for fp in files:
            f_lines = 0
            for tag, payload in iter_source_lines(fp):
                stats[tag] += 1
                f_lines += 1
                if tag != "__OK__":
                    continue
                text = payload
                if text is None:
                    stats["missing_text"] += 1
                    continue
                if len(text) < args.min_chars:
                    stats["too_short"] += 1
                    continue
                if len(text) > args.max_chars:
                    text = text[: args.max_chars]
                    stats["truncated"] += 1
                for ch in text:
                    char_classes[char_class(ch)] += 1
                line_bytes = len(text.encode("utf-8"))
                if args.limit_bytes and total_bytes + line_bytes > args.limit_bytes:
                    stats["limit_reached"] += 1
                    break
                total_bytes += line_bytes
                written += 1
                if args.format == "jsonl":
                    out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                else:
                    out.write(text + "\n")
            per_source[str(fp)] = f_lines
            if args.limit_bytes and total_bytes >= args.limit_bytes:
                break

    total_chars = sum(char_classes.values()) or 1
    manifest = {
        "input": str(src),
        "output": str(out_path),
        "format": args.format,
        "files_scanned": len(files),
        "lines_written": written,
        "bytes_written": total_bytes,
        "skipped": {
            "bad_json": stats.get("__BAD_JSON__", 0),
            "missing_text": stats.get("missing_text", 0),
            "too_short": stats.get("too_short", 0),
            "truncated": stats.get("truncated", 0),
            "bad_file": stats.get("__BAD_FILE__", 0),
        },
        "char_class_ratio": {k: round(v / total_chars, 6) for k, v in char_classes.items()},
        "per_source_lines": per_source,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("== prepare_corpus 完成 ==")
    print(f"  输出: {out_path} [{args.format}]")
    print(f"  行数 {written} | 字节 {total_bytes / 1e9:.2f} GB | 文件 {len(files)}")
    print(f"  跳过: {dict(manifest['skipped'])}")
    print(f"  字符分布: {manifest['char_class_ratio']}")
    print("  ⚠️ 若 bad_file > 0，检查源文件编码（本脚本只接受严格 UTF-8）")

    # R4-24：退出码必须反映"到底有没有产出"。原实现无论全部文件解码失败、
    # 全部行被过滤，还是一行没写，都返回 0 —— 上游 run_all.sh / CI 只看到
    # 绿色，后续训练在**空语料**上跑，训练也能"成功"产出一个纯填词的模型。
    # 这里把"没产出"变成硬失败（产物仍保留，便于查看 manifests）。
    failed_all = (
        len(files) > 0
        and stats.get("__BAD_FILE__", 0) >= len(files)
        and written == 0
    )
    if written == 0:
        print(
            f"❌ 未写出任何行（扫描 {len(files)} 个文件，"
            f"bad_file={stats.get('__BAD_FILE__', 0)}）：输出语料为空，"
            "训练/评估都会失去意义，按失败处理。",
            file=sys.stderr,
        )
        if failed_all:
            print("   全部源文件均读取失败：检查编码（只接受严格 UTF-8）与路径权限。",
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

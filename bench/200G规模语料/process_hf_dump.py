#!/usr/bin/env python3
"""process_hf_dump.py — HF 原始分片 → 训练用 jsonl（对应《数据下载与预处理.md》§5.4）。

输入（可混放，也可给目录递归扫描）：
  *.parquet      fineweb-edu / finewiki / wudao（pyarrow 流式 iter_batches，不吃内存）
  *.jsonl        SkyPile / CCI3-HQ
  *.jsonl.gz     MAP-CC / ChineseWebText2.0
输出：`{"text": "..."}` 每行一条（bbpe train --format jsonl 直接可吃）

同时做"进训练前必须做"的质检（README §2.4 A）：
  - UTF-8 严格校验（非法字节行剔除并计数）
  - 字段缺失 / JSON 坏行计数（⟹ 对应 bbpe train 的 skipped_lines）
  - 过短/超长行处理（超长截断，对应预分词 chunk 强切语义）
  - 字符集分布盘点（cjk/latin/digit/emoji ⟹ 验证多源配比）

刻意不做（README §2.4 B）：繁简/全半角/大小写规范化、剔除特殊符号。

并行：--workers N 按**文件**切分到多进程（每进程独立输出 part 文件，最后按
文件顺序拼接）。单线程实测 5.3 MB/s，200GB 要 10 小时；48 核并行可压到十几分钟。

用法：
  python3 process_hf_dump.py --input /data/raw/fineweb_edu \
      --output /data/corpus/fineweb_edu.jsonl --field text \
      --report output/corpus_fineweb.json --workers 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 字符大类（与 prepare_corpus.py 同口径）
EMOJI_RANGES = ((0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0xFE00, 0xFE0F))


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


def iter_parquet(path: Path, field: str, chunk_size: int):
    """流式读 parquet：pyarrow iter_batches，单批 chunk_size 行。"""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=chunk_size):
        cols = batch.schema.names
        if field not in cols:
            raise KeyError(f"{path}: 无字段 {field!r}，实际字段 {cols}")
        for v in batch.column(field).to_pylist():
            yield v


def iter_jsonl(path: Path, field: str):
    """流式读 jsonl / jsonl.gz：每行一个 JSON，取 field。"""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield ("__BAD_JSON__", None)
                continue
            v = obj.get(field) if isinstance(obj, dict) else None
            if not isinstance(v, str):
                yield ("__MISSING_FIELD__", None)
            else:
                yield ("__OK__", v)


def iter_source(path: Path, field: str, chunk_size: int):
    """统一迭代器：产出 (tag, text)。tag ∈ OK / BAD_JSON / MISSING_FIELD / ERROR。"""
    try:
        if path.suffix == ".parquet":
            for v in iter_parquet(path, field, chunk_size):
                if not isinstance(v, str):
                    yield ("__MISSING_FIELD__", None)
                else:
                    yield ("__OK__", v)
        elif path.suffix in (".jsonl", ".gz"):
            yield from iter_jsonl(path, field)
        else:
            yield ("__SKIP_EXT__", None)
    except Exception as e:  # noqa: BLE001 单文件失败不中断整体流水线
        yield ("__ERROR__", f"{path.name}: {e}")


def process_one(task: dict) -> dict:
    """处理单个文件，把结果追加写入 task['dest']，返回统计。

    独立成模块级函数是为了能被打包进 ProcessPoolExecutor 的 worker。
    """
    fp = Path(task["file"])
    dest = Path(task["dest"])
    stats: Counter = Counter()
    chars: Counter = Counter()
    nbytes = 0
    written = 0

    with dest.open("a", encoding="utf-8") as out:
        for tag, payload in iter_source(fp, task["field"], task["chunk_size"]):
            stats[tag] += 1
            if tag != "__OK__":
                continue
            text = payload
            # UTF-8 严格校验：能走到这说明 Python 已按 utf-8 解码成功；
            # parquet 列可能含非法代理对，显式再编码一次兜底
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                stats["invalid_utf8"] += 1
                continue
            if len(text) < task["min_chars"]:
                stats["too_short"] += 1
                continue
            if len(text) > task["max_chars"]:
                text = text[: task["max_chars"]]
                stats["truncated"] += 1
            for ch in text:
                chars[char_class(ch)] += 1
            nbytes += len(text.encode("utf-8"))
            written += 1
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    return {"name": fp.name, "written": written, "bytes": nbytes,
            "stats": dict(stats), "chars": dict(chars)}


def main() -> int:
    ap = argparse.ArgumentParser(description="HF 分片 → 训练用 jsonl（流式 + 质检，支持按文件并行）")
    ap.add_argument("--input", required=True, help="文件或目录（递归）")
    ap.add_argument("--output", required=True, help="输出 jsonl 路径")
    ap.add_argument("--field", default="text", help="文本字段名（finewiki 用 text；部分源用 content）")
    ap.add_argument("--report", default=None, help="质检报告输出路径（json）")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=100_000)
    ap.add_argument("--chunk-size", type=int, default=2048, help="parquet 批大小（行）")
    ap.add_argument("--limit-files", type=int, default=None, help="最多处理几个文件（调试）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行进程数（按文件切分；1=原单线程行为）")
    args = ap.parse_args()

    src = Path(args.input)
    files = (
        sorted(p for p in src.rglob("*") if p.is_file()
               and p.suffix in (".parquet", ".jsonl", ".gz"))
        if src.is_dir()
        else [src]
    )
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        print(f"❌ 没有可处理的文件（parquet/jsonl/gz）：{src}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    char_classes = Counter()
    total_bytes = 0
    written = 0
    per_file: dict[str, dict] = {}

    base_task = {"field": args.field, "chunk_size": args.chunk_size,
                 "min_chars": args.min_chars, "max_chars": args.max_chars}

    def absorb(res: dict) -> None:
        nonlocal total_bytes, written
        stats.update(res["stats"])
        char_classes.update(res["chars"])
        total_bytes += res["bytes"]
        written += res["written"]
        per_file[res["name"]] = {"written": res["written"]}
        print(f"  {res['name']}: {res['written']} 行", flush=True)

    workers = max(1, min(args.workers, len(files)))
    if workers == 1:
        out_path.write_text("", encoding="utf-8")  # 先清空，后续 append
        for fp in files:
            absorb(process_one({**base_task, "file": str(fp), "dest": str(out_path)}))
    else:
        print(f"  并行 {workers} 进程 / {len(files)} 个文件", flush=True)
        parts: dict[str, Path] = {}
        for i, fp in enumerate(files):
            parts[fp.name] = out_path.with_name(f".{out_path.name}.part{i}")
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = {
                    pool.submit(process_one,
                                {**base_task, "file": str(fp), "dest": str(parts[fp.name])}): fp
                    for fp in files
                }
                done = 0
                for fut in as_completed(futs):
                    absorb(fut.result())
                    done += 1
                    if done % 10 == 0 or done == len(files):
                        print(f"    进度 {done}/{len(files)}", flush=True)
            # 按文件名顺序拼接，保证输出稳定（不随完成顺序变化）
            print(f"  拼接 {len(files)} 个 part …", flush=True)
            with out_path.open("wb") as out:
                for fp in files:
                    part = parts[fp.name]
                    if part.exists():
                        with part.open("rb") as f:
                            while True:
                                buf = f.read(8 << 20)
                                if not buf:
                                    break
                                out.write(buf)
                        part.unlink()
        finally:
            for p in parts.values():
                p.unlink(missing_ok=True)

    total_chars = sum(char_classes.values()) or 1
    report = {
        "input": str(src),
        "output": str(out_path),
        "field": args.field,
        "workers": workers,
        "files_processed": len(files),
        "lines_written": written,
        "bytes_written": total_bytes,
        "skipped": {
            "bad_json": stats.get("__BAD_JSON__", 0),
            "missing_field": stats.get("__MISSING_FIELD__", 0),
            "error_files": stats.get("__ERROR__", 0),
            "invalid_utf8": stats.get("invalid_utf8", 0),
            "too_short": stats.get("too_short", 0),
            "truncated": stats.get("truncated", 0),
            "skip_ext": stats.get("__SKIP_EXT__", 0),
        },
        "char_class_ratio": {k: round(v / total_chars, 6) for k, v in char_classes.items()},
        "per_file": per_file,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== process_hf_dump 完成 ==")
    print(f"  输出: {out_path} | 行数 {written} | 字节 {total_bytes/1e9:.2f} GB")
    print(f"  跳过: {report['skipped']}")
    print(f"  字符分布: {report['char_class_ratio']}")
    if report["skipped"]["error_files"]:
        print("  ⚠️ 有文件处理失败（字段名不匹配/文件损坏），检查上面日志", file=sys.stderr)

    # R4-24：原实现即使**所有**文件都失败（字段名写错是最常见的一种）也返回 0。
    # 上游据此认为"预处理完成"，训练在空语料上照样"成功"。
    # 规则：一行都没写出 ⟹ 失败；全部文件报错 ⟹ 失败（给出字段名提示）。
    if written == 0:
        print(
            f"❌ 未写出任何行（扫描 {len(files)} 个文件）：输出 jsonl 为空，按失败处理。",
            file=sys.stderr,
        )
        if report["skipped"]["error_files"] >= len(files):
            print(
                f"   全部文件处理失败：确认 --field {args.field!r} 是否存在"
                "（parquet 常见 text/content；先用 --limit-files 1 试跑）",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

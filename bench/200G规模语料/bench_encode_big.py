#!/usr/bin/env python3
"""大数据量编码基准：bbpe / tiktoken / HF tokenizers 同文本对比。

设计要点（避免把测量误差当成性能差异）：
- 按**字节区间**切分任务，每个 worker 自己 seek+read，互不重叠；
- 每块编码完只累加 token 数、立刻丢弃 ids（1GB 文本会产生上亿 id，全留住会
  把 Python 堆撑爆，测的就是内存分配而非编码）；
- 报告 bytes/tokens/吞吐；吞吐以"文本字节数 / 墙钟"计，含解码与切分开销。

用法:
  python /tmp/bench_encode_big.py --impl bbpe --model M --input F [--procs N] [--chunk-mb 8]
"""
import argparse
import json
import multiprocessing as mp
import os
import time


def cut_points(path: str, n: int) -> list[tuple[int, int]]:
    """把文件按字节均分成 n 段，每段起点对齐到行首，保证不切碎行。"""
    size = os.path.getsize(path)
    bounds = [0]
    with open(path, "rb") as f:
        for i in range(1, n):
            f.seek(size * i // n)
            f.readline()  # 丢弃半行
            bounds.append(f.tell())
    bounds.append(size)
    return [(bounds[i], bounds[i + 1]) for i in range(n) if bounds[i + 1] > bounds[i]]


def make_encoder(impl: str, model: str):
    if impl == "bbpe":
        import sys
        # 默认用仓库自带绑定目录（<repo>/python），可用 BBPE_PY 覆盖
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.environ.get("BBPE_PY") or
                        os.path.join(here, "..", "..", "python"))
        from bbpe_tokenizer import bbpe_rs
        tok = bbpe_rs.BBPETokenizer(model)

        def enc(text: str) -> int:
            return len(tok.encode_ordinary(text))
        return enc

    if impl == "tiktoken":
        import tiktoken
        enc_obj = tiktoken.get_encoding(os.path.basename(model))

        def enc(text: str) -> int:
            return len(enc_obj.encode_ordinary(text))
        return enc

    if impl == "hf":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(model)

        def enc(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False).ids)
        return enc

    raise SystemExit(f"未知 impl: {impl}")


def work(job):
    impl, model, path, start, end, chunk_bytes = job
    enc = make_encoder(impl, model)
    nbytes = ntok = 0
    with open(path, "rb") as f:
        f.seek(start)
        tail = b""
        while True:
            pos = f.tell()
            if pos >= end:
                break
            block = f.read(min(chunk_bytes, end - pos))
            if not block:
                break
            data = tail + block
            # 只在行边界切，最后一段残行留到下一轮
            cut = data.rfind(b"\n")
            if cut == -1:
                tail = data
                continue
            text = data[: cut + 1].decode("utf-8", errors="replace")
            tail = data[cut + 1:]
            nbytes += len(data[: cut + 1])
            ntok += enc(text)
        if tail:
            text = tail.decode("utf-8", errors="replace")
            nbytes += len(tail)
            ntok += enc(text)
    return nbytes, ntok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=["bbpe", "tiktoken", "hf"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()

    size = os.path.getsize(a.input)
    best = 1e9
    nbytes = ntok = 0
    for _ in range(a.repeat):
        jobs = [(a.impl, a.model, a.input, s, e, a.chunk_mb << 20)
                for s, e in cut_points(a.input, a.procs)]
        t0 = time.monotonic()
        if a.procs == 1:
            out = [work(j) for j in jobs]
        else:
            with mp.Pool(a.procs) as pool:
                out = pool.map(work, jobs)
        dt = time.monotonic() - t0
        nbytes = sum(x[0] for x in out)
        ntok = sum(x[1] for x in out)
        best = min(best, dt)

    print(f"{a.impl:9s} procs={a.procs:<2d} | {nbytes/1e9:5.2f} GB | {best:6.1f}s | "
          f"{nbytes/best/1e6:6.2f} MB/s | {ntok/best/1e6:6.2f} M tok/s | "
          f"{nbytes/ntok:5.3f} B/tok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

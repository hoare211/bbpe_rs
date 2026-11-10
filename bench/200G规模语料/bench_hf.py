#!/usr/bin/env python3
"""HuggingFace tokenizers 对照基准：训练 + 编码。

用法:
  python bench_hf.py train  <corpus.jsonl> [vocab] [out.json] [max_gb]  # 训练 BPE 并报耗时/峰值内存
  python bench_hf.py encode <model.json> <text.txt>  # 单线程编码吞吐 + token 数

输出命名（重要）：同一次实验里要训多个词表规模时，**必须显式传不同的 out 路径**，
建议 `output/hf_bpe_{vocab}.json`（如 `hf_bpe_50256.json` / `hf_bpe_150256.json`）。
2026-11-06 踩过坑：输出路径曾硬编码，50k 那次把 150k 的模型直接覆盖，导致
"150k 编码对照"无法复跑（指标已记录，但产物丢失）。

公平性约定：
- 语料、词表规模（150256 = 256 基础字节 + 150000 merges）、min_frequency 与 bbpe 一致
- 线程数用 RAYON_NUM_THREADS 控制，与 bbpe 的 taskset 核数对齐
- HF 用自己的**标准配置**（ByteLevel BPE + GPT-2 风格预分词），预分词正则与 bbpe
  的中文优化正则不同 —— 这是"各自默认最优"的比法，会在结论里标注
"""
import json
import resource
import sys
import time


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def iter_lines(path: str, max_gb: float = 0.0):
    """逐行产出 text；max_gb > 0 时只喂前 N GB（HF 内存约 67×语料，必须切片）。"""
    budget = max_gb * (1024 ** 3) if max_gb else 0.0
    used = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue
            yield t
            if budget:
                used += len(t.encode()) + 1
                if used >= budget:
                    return


def cmd_train(corpus: str, vocab: int, out: str, max_gb: float = 0.0) -> int:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    t0 = time.monotonic()
    tok.train_from_iterator(iter_lines(corpus, max_gb), trainer=trainer)
    dt = time.monotonic() - t0
    print(f"\n训练耗时 {dt:.1f}s | 峰值内存 {rss_gb():.2f} GB | 词表 {tok.get_vocab_size()}")
    tok.save(out)
    print(f"模型已保存: {out}")
    return 0


def cmd_encode(model: str, textf: str) -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(model)
    text = open(textf, encoding="utf-8").read()
    nb = len(text.encode())
    best, ntok = 1e9, 0
    for _ in range(3):
        t0 = time.monotonic()
        enc = tok.encode(text, add_special_tokens=False)
        best = min(best, time.monotonic() - t0)
        ntok = len(enc.ids)
    print(f"{nb/1e6:.1f} MB | {best:.3f}s | {nb/best/1e6:.2f} MB/s | "
          f"{ntok/1e6:.2f}M tokens | {nb/ntok:.3f} B/token | 峰值 {rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        vocab = int(sys.argv[3]) if len(sys.argv) > 3 else 150256
        out = sys.argv[4] if len(sys.argv) > 4 else "output/hf_bpe.json"
        max_gb = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        sys.exit(cmd_train(sys.argv[2], vocab, out, max_gb))
    elif mode == "encode":
        sys.exit(cmd_encode(sys.argv[2], sys.argv[3]))
    print("用法: bench_hf.py train <corpus.jsonl> [vocab] | encode <model> <text>",
          file=sys.stderr)
    sys.exit(2)

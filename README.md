# bbpe_rs

[![CI](https://github.com/hoare211/bbpe_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/hoare211/bbpe_rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A byte-level BPE tokenizer trainer and encoder written in Rust, with Python
bindings. It is built for corpora that do not fit in memory and for pipelines
that need a reproducible model artifact instead of a black box.

`bbpe_rs` trains a vocabulary from raw JSONL or plain text, exports tokenizers
at any smaller size without retraining, and encodes/decodes with a
deterministic Rust core. A model is a plain JSON file carrying its own
provenance (git commit, corpus statistics, pre-tokenizer pattern) and is
validated for self-consistency on load.

> **Status: 0.1.0 (pre-1.0).** The model format is stable; the public API may
> still change between minor releases.

中文说明见 [README.zh-CN.md](README.zh-CN.md)。

## Highlights

- **Streaming corpus reader** — JSONL or plain text is processed line by line,
  so corpus size is limited by disk, not by RAM.
- **Out-of-core training** — add `--sample-rate` to train on a random sample of
  a corpus much larger than memory, with an independent held-out split for
  acceptance checks.
- **Export without retraining** — train once at `K_max`, then `bbpe cut` to
  derive any smaller vocabulary with identical merge order.
- **Deterministic** — same corpus and parameters produce the same merges and
  token ids, independent of thread count.
- **Reproducible artifacts** — each model records the build commit, corpus
  counters, sampling parameters and pre-tokenizer pattern; `bbpe inspect`
  re-checks them and reports failures with a non-zero exit code.
- **Python bindings** — training and long encodes release the GIL, support
  `Ctrl+C`, `CancellationToken` and `timeout`.
- **No `unsafe`** in `src/`.

## Quick start

Requirements: Rust 1.85+ (edition 2024). Python 3.8+ only for the bindings.

```bash
git clone https://github.com/hoare211/bbpe_rs.git
cd bbpe_rs

# CLI only: does not pull in pyo3, builds faster
cargo build --release --no-default-features
```

Create a tiny corpus and train a vocabulary:

```bash
mkdir -p data output
for _ in $(seq 1 100); do
  printf '{"text": "hello world 你好世界"}\n{"text": "byte pair encoding merges frequent pairs"}\n{"text": "分词器需要足够多样的语料"}\n'
done > data/train.jsonl

./target/release/bbpe train \
    --input data/train.jsonl --format jsonl \
    --vocab 300 --output output/vocab.json

./target/release/bbpe inspect output/vocab.json
./target/release/bbpe encode --model output/vocab.json --text 'hello world'
./target/release/bbpe decode --model output/vocab.json --ids 104,101,108   # hel
```

`--vocab 300` means 256 base bytes plus 44 merges. The three repeated lines
above are only a smoke-test corpus; train on your own data for real use.

The same commands are available as standalone binaries (`train`, `inspect`,
`encode`, `decode`, …) and as `bbpe <subcommand>`.

### Large corpora

```bash
./target/release/bbpe train \
    --input data/large.jsonl --format jsonl \
    --vocab 100256 --output output/vocab_100k.json \
    --sample-rate 0.2 --heldout-rate 0.005 --heldout-out output/heldout.jsonl
```

Without `--sample-rate` the trainer runs an exact full-corpus pass. With it, a
line-level random sample is used and the held-out split is written separately.
Peak RSS is approximately `9.2 × training-corpus bytes`; the trainer performs a
memory pre-check and refuses to start with a suggested down-sampling command
instead of running into OOM.

### Python

```bash
pip install maturin
maturin develop --release        # or: pip install .
```

```python
from bbpe_tokenizer import BBPETokenizer, BBPETrainer, TrainerConfig, JsonlFileReader

# Train
trainer = BBPETrainer(TrainerConfig(vocab_size=100256))
trainer.train(JsonlFileReader.static_text("data/train.jsonl"))
trainer.save("output/vocab.json")

# Encode / decode
tok = BBPETokenizer("output/vocab.json")
ids = tok.encode_ordinary("hello world")
print(ids, tok.decode(ids))
```

`torch` and `transformers` are optional and only needed for
`UTF8StrictLogitsProcessor`:

```bash
pip install "bbpe_tokenizer[logits]"
```

## Performance

Indicative numbers from a single 43 GB RAM machine with an SSD, Chinese
web text, 100K vocabulary. They are measurements, not guarantees.

| Workload | Throughput |
|---|---|
| Training (8.28 GB JSONL corpus) | ~3.7 MB/s |
| Corpus sampling | ~56 MB/s |
| Peak RSS during training | ≈ 9.2 × corpus bytes |

Run the bundled benchmark to measure your own hardware:

```bash
cargo run --release --bin bench -- data/corpus.jsonl 2256 2
```

## Model format

A model is a single JSON document (`format = "bbpe-rs-model"`,
`format_version = 1`) containing the byte vocabulary, the ordered merge list,
the pre-tokenizer pattern, special tokens and a `meta` block. Legacy v0 models
(without `format`/`pre_tokenizer_pattern`) can be migrated:

```bash
bbpe inspect legacy_tokenizer.json --migrate vocab_v1.json --pattern "<training pattern>"
```

## Project layout

```text
src/            Rust library, CLI dispatch and subcommands
python/         Python package (thin wrapper over the Rust extension)
tests/          Rust integration tests and pytest suites
bench/          large-corpus preparation and encoding benchmarks
examples/       small Rust probes
docs/           usage manual (Chinese)
```

## Documentation

- [docs/使用说明.md](docs/使用说明.md) — full usage manual: CLI reference,
  Python API, model contract, troubleshooting (Chinese).
- [README.zh-CN.md](README.zh-CN.md) — Chinese README.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. This is the usual Rust dual-license arrangement: you may pick
whichever license suits your project.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

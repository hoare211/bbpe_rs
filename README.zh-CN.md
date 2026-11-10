# bbpe_rs

[![CI](https://github.com/hoare211/bbpe_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/hoare211/bbpe_rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#许可证)

用 Rust 实现的 byte-level BPE 分词器训练器与编码器，带 Python 绑定。为
**装不进内存的语料**和**需要可复现模型产物**的流水线而设计。

`bbpe_rs` 从原始 JSONL 或纯文本训练词表，可在不重训的前提下导出任意更小规模
的分词器，并用确定性的 Rust 内核完成编解码。模型是一个普通 JSON 文件，自带
出身信息（构建 commit、语料统计、预分词 pattern），加载时做自洽性校验。

> **状态：0.1.0（pre-1.0）。** 模型格式已稳定；公开 API 仍可能在小版本间调整。

English README: [README.md](README.md)。

## 特性

- **流式语料读取** —— JSONL / 纯文本逐行处理，语料规模由磁盘而非内存决定。
- **超内存训练** —— 加 `--sample-rate` 对远大于内存的语料做行级随机采样训练，
  并生成独立 held-out 集用于验收。
- **免重训导出** —— 一次训练到 `K_max`，再用 `bbpe cut` 导出任意更小词表，
  merge 顺序完全一致。
- **确定性** —— 相同语料与参数产出相同 merges 与 token id，与线程数无关。
- **可复现产物** —— 模型记录构建 commit、语料计数、采样参数与预分词 pattern；
  `bbpe inspect` 会重新校验，失败时以非零退出码结束。
- **Python 绑定** —— 训练与长文本编码释放 GIL，支持 `Ctrl+C`、
  `CancellationToken` 与 `timeout`。
- **`src/` 零 `unsafe`**。

## 快速上手

环境要求：Rust 1.85+（edition 2024）；只有 Python 绑定需要 Python 3.8+。

```bash
git clone https://github.com/hoare211/bbpe_rs.git
cd bbpe_rs

# 纯 CLI：不拉 pyo3，编译更快
cargo build --release --no-default-features
```

建一个小语料并训练：

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

`--vocab 300` 表示 256 个基础字节 + 44 条 merge。上面重复三次的句子只是冒烟
语料，真实训练请换成自己的数据。

每个子命令也可作为独立 bin 调用（`train`、`inspect`、`encode`、`decode` …），
或统一用 `bbpe <子命令>`。

### 大语料

```bash
./target/release/bbpe train \
    --input data/large.jsonl --format jsonl \
    --vocab 100256 --output output/vocab_100k.json \
    --sample-rate 0.2 --heldout-rate 0.005 --heldout-out output/heldout.jsonl
```

不传 `--sample-rate` 即全量精确训练；传了则按行随机采样，并把 held-out 单独落盘。
训练峰值 RSS 约为 `9.2 × 训练语料字节数`。开工前会做内存预检，超预算时直接给出
可复制的降采样命令，而不是硬扛到 OOM。

### Python

```bash
pip install maturin
maturin develop --release        # 或：pip install .
```

```python
from bbpe_tokenizer import BBPETokenizer, BBPETrainer, TrainerConfig, JsonlFileReader

# 训练
trainer = BBPETrainer(TrainerConfig(vocab_size=100256))
trainer.train(JsonlFileReader.static_text("data/train.jsonl"))
trainer.save("output/vocab.json")

# 编码 / 解码
tok = BBPETokenizer("output/vocab.json")
ids = tok.encode_ordinary("hello world")
print(ids, tok.decode(ids))
```

`torch` / `transformers` 是可选依赖，仅 `UTF8StrictLogitsProcessor` 需要：

```bash
pip install "bbpe_tokenizer[logits]"
```

## 性能

以下为单机 43 GB 内存 + SSD、中文网页语料、10 万词表下的实测值，仅供参考，
不构成性能承诺。

| 场景 | 吞吐 |
|---|---|
| 训练（8.28 GB JSONL 语料） | 约 3.7 MB/s |
| 语料采样 | 约 56 MB/s |
| 训练峰值 RSS | 约 9.2 × 语料字节数 |

在自己的机器上复测：

```bash
cargo run --release --bin bench -- data/corpus.jsonl 2256 2
```

## 模型格式

模型是单个 JSON 文档（`format = "bbpe-rs-model"`、`format_version = 1`），包含
字节词表、有序 merge 列表、预分词 pattern、special tokens 与 `meta` 块。缺少
`format` / `pre_tokenizer_pattern` 的 v0 旧模型可迁移：

```bash
bbpe inspect legacy_tokenizer.json --migrate vocab_v1.json --pattern "<训练时使用的 pattern>"
```

## 目录结构

```text
src/            Rust 库、CLI 分发与各子命令
python/         Python 包（Rust 扩展的薄封装）
tests/          Rust 集成测试与 pytest 用例
bench/          大语料准备与编码基准
examples/       小型 Rust 探针
docs/           使用说明（中文）
```

## 文档

- [docs/使用说明.md](docs/使用说明.md) —— 完整使用说明：CLI 参考、Python API、
  模型契约、故障排查。
- [README.md](README.md) —— 英文 README。

## 参与贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。安全漏洞报告见 [SECURITY.md](SECURITY.md)。

## 许可证

本项目采用双许可，你可任选其一：

- Apache License 2.0（[LICENSE-APACHE](LICENSE-APACHE)）
- MIT（[LICENSE-MIT](LICENSE-MIT)）

这是 Rust 生态的常见做法：选择最适合你项目的许可证即可。

除非你明确声明，否则按 Apache-2.0 的定义有意提交以纳入本项目的任何贡献，
都将按上述双许可授权，不附加任何额外条款。

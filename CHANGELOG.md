# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-10

First public release.

### Added

- Rust library and `bbpe` CLI with subcommands `train`, `sweep`, `cut`,
  `sample`, `compare`, `inspect`, `encode`, `decode`.
- Byte-level BPE training over JSONL and plain-text corpora with a streaming,
  bounded-memory line reader.
- Sampling-based training for out-of-core corpora, with held-out split output
  and a pre-flight memory budget check.
- `cut`: derive any smaller vocabulary from a `K_max` model without retraining.
- Deterministic merges and token ids, independent of thread count.
- Model contract v1 (`format = "bbpe-rs-model"`, `format_version = 1`) with
  provenance metadata (git commit, corpus counters, sampling parameters,
  pre-tokenizer pattern) and on-load self-consistency validation.
- v0 → v1 model migration via `bbpe inspect --migrate`.
- Python package `bbpe_tokenizer` with `BBPETokenizer`, `BBPETrainer`,
  `TokenizerGlobalConfig`, `BBTokenizer`, corpus readers and
  `UTF8StrictLogitsProcessor` (optional, requires `torch` + `transformers`).
- GIL-released training and encoding, with `Ctrl+C`, `CancellationToken` and
  `timeout` support.
- Legacy command aliases (`train_vocab`, `sweep_vocab`, `cut_vocab`,
  `sample_corpus`, `inspect_model`, `compare_merges`); deprecated and slated
  for removal in a future major release.

### Known limitations

- Throughput figures are measured on Chinese web text; English- or
  code-heavy corpora should be re-benchmarked.
- Peak memory is modelled as `9.2 × corpus bytes`; for corpora far above the
  validated 8.28 GB, run a sampled pilot before a full job.
- `encode` accepts `&str` only (matching `tiktoken`); byte-level input is not
  exposed directly.
- CLI exit status is `0` when every corpus line is skipped; check the reported
  counters when a run produces a degenerate model.

[Unreleased]: https://github.com/hoare211/bbpe_rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hoare211/bbpe_rs/releases/tag/v0.1.0

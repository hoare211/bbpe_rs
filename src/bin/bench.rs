// src/bin/bench.rs
//
// 性能基准：度量峰值内存(RSS)、各阶段/总耗时、压缩率。
// 用于比对不同实现版本的性能改进效果。
//
// 用法:
//   cargo run --release --bin bench -- [corpus] [vocab] [min_freq]
//   默认: data/corpus.jsonl 2256 2
//
// 跨提交对比示例:
//   git checkout <commitA> && cargo run --release --bin bench -- > bench_A.json
//   git checkout <commitB> && cargo run --release --bin bench -- > bench_B.json
//   diff bench_A.json bench_B.json

use bbpe_rs::{BBPETokenizer, BBPETrainer, CorpusReader, RuntimeConfig, TrainerConfig};
use std::time::Instant;

/// 读取 /proc/self/status 的 VmHWM (进程峰值 RSS, 单位 KB)。Linux 专属。
fn peak_rss_kb() -> u64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmHWM:") {
                let v: Vec<&str> = line.split_whitespace().collect();
                return v.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
        }
    }
    0
}

/// m12：解析位置参数。**解析失败即报错**，不再静默回退默认值。
///
/// 原写法 `args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2256)` 会把
/// `bench corpus abc` 悄悄变成 `vocab=2256` 跑出一份看似正常、实则与意图
/// 完全不符的基准数据 —— 这类数据一旦进到跨提交对比里，误导性极强。
fn parse_positional<T>(args: &[String], idx: usize, name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match args.get(idx) {
        None => Ok(default),
        Some(s) => s
            .parse::<T>()
            .map_err(|e| format!("{name}: 无法解析 {s:?} —— {e}（解析失败不再静默回退默认值）")),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_target(false).init();

    let args: Vec<String> = std::env::args().collect();
    let corpus = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "data/corpus.jsonl".to_string());
    let vocab: usize = parse_positional(&args, 2, "vocab", 2256)?;
    let min_freq: u32 = parse_positional(&args, 3, "min_freq", 2)?;

    eprintln!(
        "[bench] corpus={}, vocab={}, min_freq={}",
        corpus, vocab, min_freq
    );

    let reader = CorpusReader::JsonlStaticText {
        filepath: corpus.clone(),
    };
    let mut config = TrainerConfig::new(vocab);
    config.min_frequency = min_freq;
    let rt = RuntimeConfig::default();
    let mut trainer = BBPETrainer::new(config, Some(rt))?;

    let rss_before = peak_rss_kb();

    // 训练 (阶段耗时由 trainer 内 [bench] 日志输出到 stderr)
    let t0 = Instant::now();
    trainer.train(&reader)?;
    let total_time = t0.elapsed().as_secs_f64();

    let rss_after = peak_rss_kb();

    // 质量度量：压缩率 (token/char, 越低压缩越好)
    std::fs::create_dir_all("output")?;
    let model_path = "output/bench_model.json";
    trainer.save(model_path)?;
    let tokenizer = BBPETokenizer::load(model_path)?;
    let test_text = "人工智能是计算机科学的一个分支，Python 是一种高级编程语言。\
                     Machine learning models learn from large amounts of data.\
                     自然语言处理让计算机能够理解人类语言。";
    let ids = tokenizer.encode_ordinary(test_text);
    let compression = ids.len() as f64 / test_text.len() as f64;
    let _ = std::fs::remove_file(model_path);

    // 结构化输出 (JSON)，便于跨提交 diff 对比
    let result = serde_json::json!({
        "corpus": corpus,
        "vocab": vocab,
        "min_freq": min_freq,
        "total_time_sec": format!("{:.3}", total_time),
        "peak_rss_kb": rss_after,
        "peak_rss_delta_kb": rss_after.saturating_sub(rss_before),
        "merges": trainer.state.merges.len(),
        "compression_token_per_char": format!("{:.4}", compression),
    });

    println!("\n=== BENCH RESULT (JSON) ===");
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

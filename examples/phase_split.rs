// 临时基准：拆分 encode 的耗时构成（预分词 vs BPE 合并）
// 用法：cargo run --release --example phase_split -- <model.json> <text.txt>
use bbpe_rs::pre_tokenizer::{DEFAULT_PATTERN, PreTokenizer};
use bbpe_rs::tokenizer::BBPETokenizer;
use std::io::Read;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).cloned().unwrap_or_default();
    let textf = args.get(2).cloned().unwrap_or_default();

    let mut text = String::new();
    std::fs::File::open(&textf)?.read_to_string(&mut text)?;
    let nbytes = text.len();

    let tok = BBPETokenizer::load(&model)?;
    let pt = PreTokenizer::new(DEFAULT_PATTERN)?;

    // 1) 仅预分词
    for _ in 0..2 {
        let t0 = Instant::now();
        let mut chunks = 0usize;
        let mut chunk_bytes = 0usize;
        for line in text.lines() {
            pt.tokenize_with_checked(line, |b, _| {
                chunks += 1;
                chunk_bytes += b.len();
            })?;
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "预分词: {dt:6.3}s  {:.2} MB/s | chunk {} | 平均 {:.1} B",
            nbytes as f64 / dt / 1e6,
            chunks,
            chunk_bytes as f64 / chunks.max(1) as f64
        );
    }

    // 1.5) chunk 重复度：命中率上限 = 1 - 唯一chunk数/总chunk数
    {
        use rustc_hash::FxHashMap;
        let mut seen: FxHashMap<&[u8], u64> = FxHashMap::default();
        let mut total = 0u64;
        for line in text.lines() {
            pt.tokenize_with_checked(line, |b, _| {
                total += 1;
                *seen.entry(b).or_insert(0) += 1;
            })?;
        }
        let uniq = seen.len() as u64;
        let mut counts: Vec<u64> = seen.values().copied().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let top10 = counts.iter().take(10).sum::<u64>();
        let top1pct = counts
            .iter()
            .take((uniq as usize / 100).max(1))
            .sum::<u64>();
        let mut b = [0u64; 4];
        for (&k, &v) in seen.iter() {
            let n = k.len();
            let idx = if n < 3 {
                0
            } else if n <= 8 {
                1
            } else if n <= 32 {
                2
            } else {
                3
            };
            b[idx] += v;
        }
        println!(
            "长度分布(按出现次数): <3B {:.1}% | 3-8B {:.1}% | 9-32B {:.1}% | >32B {:.1}%",
            100.0 * b[0] as f64 / total as f64,
            100.0 * b[1] as f64 / total as f64,
            100.0 * b[2] as f64 / total as f64,
            100.0 * b[3] as f64 / total as f64
        );
        println!(
            "chunk 重复度: 总 {total} | 唯一 {uniq} | 理想命中率上限 {:.1}% | top10 占 {:.1}% | top1% 占 {:.1}%",
            100.0 * (1.0 - uniq as f64 / total as f64),
            100.0 * top10 as f64 / total as f64,
            100.0 * top1pct as f64 / total as f64
        );
    }

    // 2) 完整编码
    for _ in 0..2 {
        let t0 = Instant::now();
        let ids = tok.encode_ordinary(&text);
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "完整 encode: {dt:6.3}s  {:.2} MB/s | tokens {}",
            nbytes as f64 / dt / 1e6,
            ids.len()
        );
    }
    Ok(())
}

// 追加：chunk 重复度统计（用于判断 chunk 级缓存/memo 的收益上限）

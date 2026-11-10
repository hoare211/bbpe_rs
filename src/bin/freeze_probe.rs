// src/bin/freeze_probe.rs
//
// P0 冻结率探针（方案 §16 P0，生死判定项）——"用最便宜的实验买最贵的信息"。
//
// 目的：在启动 M1 全量实施之前，用真实语料测出三个生死数字：
//   1. 冻结率 @ m_f ∈ {100, 1000, 10^4}（一次跑完三档）
//   2. "含高频 pair 罕见词"字节占比（R6/软肋一的直接量化：非冻结且 count ≤ 罕见阈值）
//   3. 边过滤（引理 3）前后的索引边数（F15 收益验证）
// 附带产出：pre_tokenizer 实测吞吐（P3 检查项）+ top 热点字节对。
//
// 架构 = 阶段 0 的迷你原型：
//   Phase 1（单遍流式）：语料 → 预分词 → T₀: u64[65536] 按出现累加（512KB 常驻）
//                       + SpillingBucket 词频计数（超阈值溢写 sorted run，内存受控）
//   Phase 2（归并判定）：RunMerger 逐词去重求和 → 对每个 (word, count) 查 T₀ 判定冻结
//
// 判定规则（预写死，方案 v1.1 §16 P0，套用 unique 冻结率 @ m_f=100）：
//   ≥40% → 放心进 M1；20~40% → M1 照做但预期管理改为近似模式为主；
//   <20% → 暂停 M1，先做 Q4 粒度决策或 Plan B（区间冻结）设计。
//
// 冻结率双口径（都打印）：
//   - unique 冻结率：被冻结去重词 / 去重词总数（内存视角，规则套用此口径）
//   - occ 字节冻结率：被冻结词 Σ(count×len) / 全部 Σ(count×len)（语料覆盖视角）
//
// 用法：
//   cargo run --release --bin freeze_probe -- \
//       --input /path/to/corpus.jsonl \
//       [--format jsonl|text] [--max-bytes N] [--mf 100,1000,10000] \
//       [--rare-count 10] [--flush-threshold 5000000] [--run-dir /tmp] [--keep-runs]

use bbpe_rs::external::{RunMerger, SpillingBucket, create_run_dir};
use bbpe_rs::pre_tokenizer::{DEFAULT_PATTERN, PreTokenizer};
use bbpe_rs::reader::{CorpusReader, CorpusStats};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// m11：run 目录的 RAII 守卫。
///
/// 原实现只在**正常结束**时清理 run 目录，而 `--format` 非法、SpillingBucket
/// 溢写失败等**早退路径**会直接 `return Err(..)`，把临时 run 目录（可能数 GB）
/// 遗留在 `/tmp`。这里用 Drop 保证任何退出路径都清理；需要保留时显式 `keep()`。
struct RunDirGuard(PathBuf, bool);

impl RunDirGuard {
    fn new(p: PathBuf) -> Self {
        Self(p, true)
    }
    /// 放弃自动清理（对应 `--keep-runs`）
    fn keep(mut self) {
        self.1 = false;
    }
    fn path(&self) -> &PathBuf {
        &self.0
    }
}

impl Drop for RunDirGuard {
    fn drop(&mut self) {
        if self.1 {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

struct Args {
    input: String,
    format: String,
    max_bytes: u64,
    mfs: Vec<u64>,
    rare_count: u64,
    flush_threshold: usize,
    run_dir_base: PathBuf,
    keep_runs: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        input: String::new(),
        format: "jsonl".to_string(),
        max_bytes: u64::MAX,
        mfs: vec![100, 1000, 10_000],
        rare_count: 10,
        flush_threshold: 5_000_000,
        run_dir_base: std::env::temp_dir(),
        keep_runs: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => args.input = it.next().ok_or("--input 需要参数")?,
            "--format" => args.format = it.next().ok_or("--format 需要参数")?,
            "--max-bytes" => {
                args.max_bytes = it
                    .next()
                    .ok_or("--max-bytes 需要参数")?
                    .parse()
                    .map_err(|e| format!("--max-bytes 解析失败: {e}"))?
            }
            "--mf" => {
                let s = it.next().ok_or("--mf 需要参数")?;
                args.mfs = s
                    .split(',')
                    .map(|x| x.trim().parse().map_err(|e| format!("--mf 解析失败: {e}")))
                    .collect::<Result<Vec<_>, _>>()?;
                if args.mfs.is_empty() {
                    return Err("--mf 不能为空".into());
                }
            }
            "--rare-count" => {
                args.rare_count = it
                    .next()
                    .ok_or("--rare-count 需要参数")?
                    .parse()
                    .map_err(|e| format!("--rare-count 解析失败: {e}"))?
            }
            "--flush-threshold" => {
                args.flush_threshold = it
                    .next()
                    .ok_or("--flush-threshold 需要参数")?
                    .parse()
                    .map_err(|e| format!("--flush-threshold 解析失败: {e}"))?
            }
            "--run-dir" => {
                args.run_dir_base = PathBuf::from(it.next().ok_or("--run-dir 需要参数")?)
            }
            "--keep-runs" => args.keep_runs = true,
            other => return Err(format!("未知参数: {other}")),
        }
    }
    if args.input.is_empty() {
        return Err("必须提供 --input <语料路径>".into());
    }
    // m11：阈值下限。`SpillingBucket::add` 的判定是 `map.len() >= threshold`，
    // 取 0 或 1 会导致**几乎每个词都溢写一次**，瞬间产生海量 run 文件并在
    // Phase 2 归并时打满文件描述符（EMFILE）。这里硬性拒绝 <2，并对过小值
    // 给出提示（默认 5_000_000 才是合理量级）。
    if args.flush_threshold < 2 {
        return Err(format!(
            "--flush-threshold 必须 ≥ 2（实际 {}）：阈值过小会导致每个词单独溢写一个 \
             run 文件，归并期必然耗尽文件描述符",
            args.flush_threshold
        ));
    }
    if args.flush_threshold < 1000 {
        eprintln!(
            "⚠️  --flush-threshold={} 远小于默认值 5000000，将产生大量 run 文件且归并缓慢",
            args.flush_threshold
        );
    }
    Ok(args)
}

/// 单个冻结阈值下的累计统计
#[derive(Default)]
struct MfStats {
    frozen_unique: u64,
    active_unique: u64,
    frozen_occ_bytes: u64,   // Σ count×len（冻结词）
    active_occ_bytes: u64,   // Σ count×len（活跃词）
    active_dedup_bytes: u64, // Σ len（活跃去重词 = flat_data 预算）
    rare_active_unique: u64, // 非冻结且 count ≤ rare（软肋一/R6）
    rare_active_occ_bytes: u64,
    rare_active_dedup_bytes: u64,
    edges_total: u64,     // 活跃词全部相邻 pair（无过滤，F15 炸弹口径）
    edges_kept_m100: u64, // T₀ ≥ 100 的初始边（引理 3，严格模式 m=100）
    edges_kept_m2: u64,   // T₀ ≥ 2 的初始边（近似模式取 m=2 的对照口径）
}

impl MfStats {
    fn freeze_rate_unique(&self) -> f64 {
        let t = self.frozen_unique + self.active_unique;
        if t == 0 {
            0.0
        } else {
            self.frozen_unique as f64 / t as f64
        }
    }
    fn freeze_rate_occ(&self) -> f64 {
        let t = self.frozen_occ_bytes + self.active_occ_bytes;
        if t == 0 {
            0.0
        } else {
            self.frozen_occ_bytes as f64 / t as f64
        }
    }
}

/// FNV-1a：用于采样的确定性哈希（同一词在多次运行中判定一致，保证结果可复现）
fn fnv1a(b: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &x in b {
        h ^= x as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

/// 按 T₀ 名次分桶统计"被抽走比例"的 均值±σ。
/// 分桶的必要性：合并顺序由【头部】pair 决定，尾部（T₀ 小）本就接近噪声，
/// 混在一起算总 σ 会被尾部稀释/放大，必须分开看才公平。
fn bucket_stats(cands: &[(usize, u64)], drop: &[u64]) -> String {
    let bounds: [(usize, usize, &str); 4] = [
        (0, 100, "top100"),
        (100, 1_000, "101~1K"),
        (1_000, 3_000, "1K~3K"),
        (3_000, usize::MAX, "3K+"),
    ];
    let mut out = String::new();
    for (lo, hi, nm) in bounds {
        let hi = hi.min(cands.len());
        if lo >= hi {
            out.push_str(&format!("  {nm}: n/a"));
            continue;
        }
        let n = (hi - lo) as f64;
        let mut s = 0.0f64;
        let mut sq = 0.0f64;
        for (p, t) in &cands[lo..hi] {
            let f = drop[*p] as f64 / *t as f64;
            s += f;
            sq += f * f;
        }
        let m = s / n;
        let sd = ((sq / n - m * m).max(0.0)).sqrt();
        // 注意：{nm} 是内联捕获，不消耗位置参数；两个 {:.1} 依次吃掉 m、sd
        out.push_str(&format!("  {nm}: {:.1}%±{:.1}%", m * 100.0, sd * 100.0));
    }
    out
}

fn fmt_gb(b: u64) -> String {
    format!("{:.2} GB", b as f64 / 1e9)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let pt = PreTokenizer::new(DEFAULT_PATTERN)?;
    // m11：RAII 守卫 —— 下面任何 `return Err` / `?` 都会自动清理 run 目录
    let run_dir = RunDirGuard::new(create_run_dir(&args.run_dir_base)?);

    let reader = match args.format.as_str() {
        "jsonl" => CorpusReader::JsonlStaticText {
            filepath: args.input.clone(),
        },
        "text" => CorpusReader::Text {
            filepath: args.input.clone(),
        },
        other => {
            return Err(format!("未知 --format: {other}（支持 jsonl | text）").into());
        }
    };

    println!("== freeze_probe（P0 生死判定项） ==");
    println!("语料: {} [format={}]", args.input, args.format);
    println!(
        "m_f 扫描: {:?} | 罕见词阈值 count≤{} | 溢写阈值 {} 条 | runs 目录 {}",
        args.mfs,
        args.rare_count,
        args.flush_threshold,
        run_dir.path().display()
    );

    // ================= Phase 1：单遍流式统计（阶段 0 迷你原型） =================
    let mut t0: Vec<u64> = vec![0; 65536]; // 全量精确初始 pair 频次表（512KB）
    let mut bucket = SpillingBucket::new(run_dir.path().clone(), 0, args.flush_threshold);
    let mut bytes_read: u64 = 0;
    let mut occurrences: u64 = 0;
    let mut single_byte_occ: u64 = 0;
    let mut next_progress: u64 = 512 * 1024 * 1024;
    let mut io_err: Option<std::io::Error> = None;

    let t_start = Instant::now();
    // m11：改用带计数的迭代 —— 原 `iter_lines()` 会把 JSONL 坏行/缺字段行静默
    // 跳过、把非法 UTF-8 静默替换成 U+FFFD，而本探针恰恰是「用真实语料测冻结率」，
    // 语料被改动多少必须可见，否则结论建立在看不见的偏移上。
    let corpus_stats = Arc::new(CorpusStats::default());
    for line in reader.iter_lines_with_stats(corpus_stats.clone())? {
        bytes_read += line.len() as u64;
        if bytes_read > args.max_bytes {
            println!("[截断] 已达 --max-bytes 上限，停止读取");
            break;
        }
        pt.tokenize_with(&line, |w, _| {
            occurrences += 1;
            if w.len() >= 2 {
                for i in 0..w.len() - 1 {
                    t0[((w[i] as usize) << 8) | (w[i + 1] as usize)] += 1;
                }
            } else {
                single_byte_occ += 1;
            }
            if io_err.is_none() {
                if let Err(e) = bucket.add(w) {
                    io_err = Some(e);
                }
            }
        });
        if let Some(e) = io_err.take() {
            return Err(format!("SpillingBucket 溢写失败（修复 1 生效，训练终止）: {e}").into());
        }
        if bytes_read >= next_progress {
            let secs = t_start.elapsed().as_secs_f64();
            println!(
                "  ... {} 读入 | {:.0} MB/s | word 出现 {}",
                fmt_gb(bytes_read),
                bytes_read as f64 / 1e6 / secs,
                occurrences
            );
            next_progress += 512 * 1024 * 1024;
        }
    }
    bucket.finish()?;
    let phase1_secs = t_start.elapsed().as_secs_f64();

    // ================= Phase 2：归并 + 冻结判定 =================
    let mut stats: Vec<MfStats> = args.mfs.iter().map(|_| MfStats::default()).collect();
    let mut total_unique: u64 = 0;
    let mut single_unique: u64 = 0;

    let t2_start = Instant::now();
    // ---- Plan B（区间冻结）量化：K 档"词频 ≤ K 全冻结"，测内存收益 vs 精度损失 ----
    let ks: Vec<u64> = vec![1, 2, 5, 10];
    let mut removed: Vec<Vec<u64>> = ks.iter().map(|_| vec![0u64; 65536]).collect();
    let mut pb_frozen_unique = vec![0u64; ks.len()];
    let mut pb_frozen_occ = vec![0u64; ks.len()];
    let mut pb_frozen_dedup = vec![0u64; ks.len()];
    // 轨 A 采样对照：每个去重词以概率 s 独立保留（整词全留 → 方差偏大，属保守/悲观估计）
    let sample_permille: u32 = 50; // 5%
    let mut kept_sample: Vec<u64> = vec![0u64; 65536];
    let mut total_words = 0u64; // 长度 ≥ 2 的去重词（活跃集候选全集）
    let mut total_dedup_bytes = 0u64;
    let mut total_occ_bytes = 0u64;
    {
        let rare = args.rare_count;
        let mfs = &args.mfs;
        let t0_ref = &t0;
        let ks_ref = &ks;
        #[allow(clippy::needless_range_loop)]
        let mut consume = |w: &[u8], count: u64| {
            total_unique += 1;
            if w.len() < 2 {
                single_unique += 1; // 单字节词：阶段 1 直接丢弃，不占活跃集（方案 §7.1）
                return;
            }
            total_words += 1;
            total_dedup_bytes += w.len() as u64;
            let occ_bytes = count * w.len() as u64;
            total_occ_bytes += occ_bytes;
            // Plan B：该词被哪些 K 档冻结（K ≥ count）
            let ks_start = ks_ref.partition_point(|&k| k < count);
            // 轨 A 采样：整词独立保留（保守估计，见 sample_permille 注释）
            let keep_sample = fnv1a(w) % 1000 < sample_permille;
            // 一遍扫描词内 pair：最热频次 + 双阈值边计数 + Plan B 移除频次 + 采样保留频次
            let mut hot: u64 = 0;
            let mut kept100: u64 = 0;
            let mut kept2: u64 = 0;
            for i in 0..w.len() - 1 {
                let p = ((w[i] as usize) << 8) | (w[i + 1] as usize);
                let v = t0_ref[p];
                if v > hot {
                    hot = v;
                }
                if v >= 100 {
                    kept100 += 1;
                }
                if v >= 2 {
                    kept2 += 1;
                }
                // clippy::needless_range_loop 在这里是**误报**：`k` 同时索引
                // `removed` / `pb_frozen_*` 多个数组，改成迭代器只会把三个数组
                // 强行 zip 起来（可读性显著变差），故显式豁免并说明理由。
                for k in ks_start..ks_ref.len() {
                    removed[k][p] += count;
                }
                if keep_sample {
                    kept_sample[p] += count;
                }
            }
            if ks_start < ks_ref.len() {
                // 同上：`k` 同时索引三个数组，迭代器化反而更绕，显式豁免。
                for k in ks_start..ks_ref.len() {
                    pb_frozen_unique[k] += 1;
                    pb_frozen_occ[k] += occ_bytes;
                    pb_frozen_dedup[k] += w.len() as u64;
                }
            }
            let npairs = (w.len() - 1) as u64;
            let is_rare = count <= rare;
            for (idx, mf) in mfs.iter().enumerate() {
                let s = &mut stats[idx];
                if hot < *mf {
                    s.frozen_unique += 1;
                    s.frozen_occ_bytes += occ_bytes;
                } else {
                    s.active_unique += 1;
                    s.active_occ_bytes += occ_bytes;
                    s.active_dedup_bytes += w.len() as u64;
                    s.edges_total += npairs;
                    s.edges_kept_m100 += kept100;
                    s.edges_kept_m2 += kept2;
                    if is_rare {
                        s.rare_active_unique += 1;
                        s.rare_active_occ_bytes += occ_bytes;
                        s.rare_active_dedup_bytes += w.len() as u64;
                    }
                }
            }
        };

        let run_paths: Vec<PathBuf> = bucket.run_paths().to_vec();
        if run_paths.is_empty() {
            // 小语料路径：从未溢写，直接遍历内存 map（零 IO）
            bucket.for_each_word(|w, c| consume(w, c))?;
        } else {
            let mut merger = RunMerger::new(&run_paths)?;
            while let Some((w, c)) = merger.next_word()? {
                consume(&w, c);
            }
        }
    }
    let phase2_secs = t2_start.elapsed().as_secs_f64();

    // ================= 报告 =================
    println!();
    println!("== Phase 1 结果（阶段 0 迷你原型） ==");
    println!("  读入 ~{}（行字节和，不含换行）", fmt_gb(bytes_read));
    println!(
        "  word 出现 {} 次（单字节 {} 次）| 去重词 {}（单字节 {} 个）",
        occurrences, single_byte_occ, total_unique, single_unique
    );
    println!(
        "  耗时 {:.1}s | 吞吐 {:.0} MB/s（P3 参考值，单线程）| sorted runs {} 个",
        phase1_secs,
        bytes_read as f64 / 1e6 / phase1_secs,
        bucket.run_paths().len()
    );
    println!("  Phase 2 归并判定耗时 {:.1}s", phase2_secs);
    println!();

    println!("== Phase 2 冻结判定（T₀ 全量精确，定理 3） ==");
    for (idx, mf) in args.mfs.iter().enumerate() {
        let s = &stats[idx];
        let edges_per_word = if s.active_unique > 0 {
            s.edges_kept_m100 as f64 / s.active_unique as f64
        } else {
            0.0
        };
        // 活跃集内存估算（方案 §11 口径，不含动态边 1~4GB 与 A 表 <1GB）
        let est_mem = s.active_dedup_bytes
            + 16 * s.active_unique // starts(u32)+lens(u32)+weights(u64)
            + s.edges_kept_m100 * 8;
        println!("  ---- m_f = {} ----", mf);
        println!(
            "    冻结率: unique {:.1}% | occ 字节 {:.1}%",
            s.freeze_rate_unique() * 100.0,
            s.freeze_rate_occ() * 100.0
        );
        println!(
            "    活跃去重词 {} | flat_data {} | 活跃 Σ(count×len) {}",
            s.active_unique,
            fmt_gb(s.active_dedup_bytes),
            fmt_gb(s.active_occ_bytes)
        );
        println!(
            "    索引边: 无过滤 {}（{}）| 引理3过滤(m=100) {}（{}，{:.1} 边/词）| m=2 口径 {}（{}）",
            s.edges_total,
            fmt_gb(s.edges_total * 8),
            s.edges_kept_m100,
            fmt_gb(s.edges_kept_m100 * 8),
            edges_per_word,
            s.edges_kept_m2,
            fmt_gb(s.edges_kept_m2 * 8)
        );
        println!("    活跃集内存估算（不含动态边/A 表）: {}", fmt_gb(est_mem));
        println!(
            "    [R6/软肋一] 罕见活跃词（非冻结且 count≤{}）: {} 个 | occ 字节占比 {:.2}% | dedup 字节 {}",
            args.rare_count,
            s.rare_active_unique,
            if s.active_occ_bytes + s.frozen_occ_bytes > 0 {
                s.rare_active_occ_bytes as f64 / (s.active_occ_bytes + s.frozen_occ_bytes) as f64
                    * 100.0
            } else {
                0.0
            },
            fmt_gb(s.rare_active_dedup_bytes)
        );
    }
    println!();

    // ================= Plan B（区间冻结）量化 =================
    // 与"定理 2 严格冻结"相对：这里【不论 pair 内容】，按词频区间直接冻结 count ≤ K 的词。
    // 收益看内存，代价看两件事：
    //   硬损失 = 冻结后 T₀ 掉到 m_f 以下、合并被彻底丢掉的候选 pair 数；
    //   软风险 = 候选 pair 频次被抽走的比例（可能改变贪心 argmax 顺序 → 合并序列不同）。
    let pb_mf: u64 = if args.mfs.contains(&100) {
        100
    } else {
        args.mfs[0]
    };
    println!("== Plan B 量化：区间冻结（count≤K 全冻结，不论 pair 内容） ==");
    println!("  参考阈值 m_f={pb_mf}（候选 pair = T₀ ≥ m_f）| 去重词全集 {total_words}");
    println!(
        "  {:>4} {:>9} {:>9} {:>9} {:>14} {:>9} {:>9} {:>8}",
        "K",
        "冻结去重词",
        "dedup省",
        "occ丢",
        "候选pair丢失",
        "最大相对降",
        "平均相对降",
        "相对降σ"
    );
    for (k_idx, k) in ks.iter().enumerate() {
        let mut n_cand = 0u64;
        let mut pairs_lost = 0u64;
        let mut max_rel = 0.0f64;
        let mut sum_rel = 0.0f64;
        let mut sum_sq = 0.0f64;
        for p in 0..65536usize {
            let t = t0[p];
            if t < pb_mf {
                continue;
            }
            n_cand += 1;
            let r = removed[k_idx][p];
            if t.saturating_sub(r) < pb_mf {
                pairs_lost += 1;
            }
            let rel = r as f64 / t as f64;
            if rel > max_rel {
                max_rel = rel;
            }
            sum_rel += rel;
            sum_sq += rel * rel;
        }
        let avg_rel = if n_cand > 0 {
            sum_rel / n_cand as f64
        } else {
            0.0
        };
        // σ = 各候选 pair 被抽走比例的离散度 —— 这才是"合并顺序是否被扰乱"的真指标：
        // 均值只决定整体尺度（可等效缩放），σ 决定 pair 之间的相对次序是否翻转。
        let var = if n_cand > 1 {
            (sum_sq / n_cand as f64 - avg_rel * avg_rel).max(0.0)
        } else {
            0.0
        };
        println!(
            "  {:>4} {:>9} {:>9} {:>9} {:>14} {:>9} {:>9} {:>8}",
            k,
            format!(
                "{:.1}%",
                pb_frozen_unique[k_idx] as f64 / total_words.max(1) as f64 * 100.0
            ),
            format!(
                "{:.1}%",
                pb_frozen_dedup[k_idx] as f64 / total_dedup_bytes.max(1) as f64 * 100.0
            ),
            format!(
                "{:.1}%",
                pb_frozen_occ[k_idx] as f64 / total_occ_bytes.max(1) as f64 * 100.0
            ),
            format!("{} / {}", pairs_lost, n_cand),
            format!("{:.1}%", max_rel * 100.0),
            format!("{:.2}%", avg_rel * 100.0),
            format!("{:.2}%", var.sqrt() * 100.0)
        );
    }
    println!("  读法：『候选pair丢失』= 硬损失（该 pair 的合并彻底不会发生）；");
    println!("        『相对降』= 软风险（频次被抽走比例，比例越接近 0 合并顺序越稳）。");
    println!();

    // top-20 热点字节对（调试信息）
    let mut top: Vec<(usize, u64)> = t0
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > 0)
        .map(|(i, v)| (i, *v))
        .collect();
    top.sort_unstable_by_key(|&(_, v)| std::cmp::Reverse(v));
    println!("== top-20 热点字节对（T₀） ==");
    for (i, v) in top.iter().take(20) {
        let (a, b) = ((i >> 8) as u8, (i & 0xff) as u8);
        let pair = [a, b];
        let disp = String::from_utf8_lossy(&pair);
        println!("    {:02x} {:02x} ({:>4}) : {}", a, b, disp, v);
    }
    println!();

    // ================= 候选池规模 + 轨 A 采样对照 =================
    let t_at = |rank: usize| -> u64 { top.get(rank - 1).map(|(_, v)| *v).unwrap_or(0) };
    println!("== 候选池规模与采样鲁棒性（轨 A 对照） ==");
    println!(
        "  非零 T₀ pair 数 {} | 第 1/100/1K/10K/32K 名 T₀ = {}/{}/{}/{}/{}",
        top.len(),
        t_at(1),
        t_at(100),
        t_at(1_000),
        t_at(10_000),
        t_at(32_000)
    );
    let mut n_cand_all = 0u64;
    let mut samp_lost = 0u64;
    let mut samp_sum = 0.0f64;
    let mut samp_sq = 0.0f64;
    for p in 0..65536usize {
        let t = t0[p];
        if t < pb_mf {
            continue;
        }
        n_cand_all += 1;
        let kept = kept_sample[p];
        if kept < pb_mf {
            samp_lost += 1;
        }
        let rel = 1.0 - (kept as f64 / t as f64);
        samp_sum += rel;
        samp_sq += rel * rel;
    }
    let samp_avg = if n_cand_all > 0 {
        samp_sum / n_cand_all as f64
    } else {
        0.0
    };
    let samp_sd = if n_cand_all > 1 {
        (samp_sq / n_cand_all as f64 - samp_avg * samp_avg)
            .max(0.0)
            .sqrt()
    } else {
        0.0
    };
    let survived = n_cand_all - samp_lost;
    println!(
        "  采样 {}‰（整词独立保留，保守上界）: 候选 pair 平均移除 {:.2}% | σ {:.2}% | 丢失 {} / {}",
        sample_permille,
        samp_avg * 100.0,
        samp_sd * 100.0,
        samp_lost,
        n_cand_all
    );
    println!(
        "  采样后存活候选 pair {} 个 → {}",
        survived,
        if survived >= 32_000 {
            "初始池 ≥ 32K：初始候选即已覆盖 vocab_size（推论 6/7）"
        } else {
            "初始池 < 32K 属正常：vocab 上限不受初始池约束（合并会动态产生新 pair）"
        }
    );
    println!("  【关键对比】Plan B 与采样都在删词，但性质相反：");
    println!("    Plan B：按词频偏置删除 → 每个 pair 被抽走的比例差异极大（σ 大）→ 相对次序翻转；");
    println!("    采样  ：无偏删除 → 所有 pair 同期望比例缩放（σ 仅来自抽样方差）→ 头部次序稳定。");
    println!();

    // ---- 分桶对比：失真究竟集中在【头部】（决定合并顺序）还是【尾部】（噪声）----
    let mut cands: Vec<(usize, u64)> = t0
        .iter()
        .enumerate()
        .filter(|(_, v)| **v >= pb_mf)
        .map(|(i, v)| (i, *v))
        .collect();
    cands.sort_unstable_by_key(|&(_, v)| std::cmp::Reverse(v));
    let k10 = ks.iter().position(|&k| k == 10).unwrap_or(ks.len() - 1);
    let mut drop_pb1 = vec![0u64; 65536];
    let mut drop_pb10 = vec![0u64; 65536];
    let mut drop_samp = vec![0u64; 65536];
    for p in 0..65536usize {
        drop_pb1[p] = removed[0][p];
        drop_pb10[p] = removed[k10][p];
        drop_samp[p] = t0[p].saturating_sub(kept_sample[p]);
    }
    println!("  == 失真分布：各 T₀ 名次区间的『被抽走比例』均值±σ ==");
    println!("  Plan B K=1 :{}", bucket_stats(&cands, &drop_pb1));
    println!("  Plan B K=10:{}", bucket_stats(&cands, &drop_pb10));
    println!("  采样 5%    :{}", bucket_stats(&cands, &drop_samp));
    println!(
        "  判读：合并顺序由【头部】pair 决定。Plan B 在头部即有大比例抽走且 σ 大 → 顺序必乱；"
    );
    println!("        采样的失真集中在尾部（T₀ 小、接近噪声），头部近似等比例缩放 → 顺序稳。");
    println!();

    // P0 判定（预写死规则，套用 unique 冻结率 @ m_f=100 或首个 m_f）
    let probe_mf = args.mfs[0];
    let verdict_mf = if args.mfs.contains(&100) {
        100
    } else {
        probe_mf
    };
    let s = &stats[args.mfs.iter().position(|m| *m == verdict_mf).unwrap()];
    let rate = s.freeze_rate_unique();
    println!(
        "== P0 判定（规则预写死于方案 v1.1 §16；unique 冻结率 @ m_f={} = {:.1}%） ==",
        verdict_mf,
        rate * 100.0
    );
    if rate >= 0.40 {
        println!("  [✅ 放心进 M1] 冻结率 ≥ 40%：严格模式预期可行，按 §14 路线推进。");
    } else if rate >= 0.20 {
        println!("  [⚠️  M1 照做，预期管理改为近似模式为主] 冻结率 20~40%：守门大概率触发，");
        println!("      按 §9.1 升级 m_f，偏离由定理 5/6 + 证书界定。");
    } else {
        println!("  [🛑 暂停 M1] 冻结率 < 20%：软肋一/R6 结构性暴露，先做 Q4 粒度决策");
        println!("      （中文句段级粒度是结构性压制因素）或 Plan B（区间冻结）设计。");
    }
    println!(
        "  参考：m_f=1000 时 unique 冻结率 {} | m_f=10^4 时 {}（守门升级的预期收益）",
        args.mfs
            .iter()
            .position(|m| *m == 1000)
            .map(|i| format!("{:.1}%", stats[i].freeze_rate_unique() * 100.0))
            .unwrap_or_else(|| "未测".into()),
        args.mfs
            .iter()
            .position(|m| *m == 10_000)
            .map(|i| format!("{:.1}%", stats[i].freeze_rate_unique() * 100.0))
            .unwrap_or_else(|| "未测".into())
    );
    println!();
    println!(
        "注：内存估算为初始边口径；动态边（新 pair，不可过滤）另计 ~1-4GB；\n    \
         A 表/堆/T₀ < 1GB。近似模式若取 m=2，初始边过滤退化至 m=2 口径（见上表对照）——\n    \
         依据推论 7（vocab_size 先到时 m 不影响输出），近似模式建议同样保持 m=100。"
    );

    // m11：输入侧静默行为必须可见（与 cli/train 同口径）
    let (total_lines, total_bytes, skipped, replaced) = corpus_stats.snapshot();
    println!(
        "语料读取: {} 行 | {} | skipped_lines {} | replacement_chars {}",
        total_lines,
        fmt_gb(total_bytes),
        skipped,
        replaced
    );
    if skipped > 0 {
        println!(
            "  ⚠️  {} 行被静默跳过（JSONL 坏行 / 缺字段），占比 {:.4}%",
            skipped,
            skipped as f64 / total_lines.max(1) as f64 * 100.0
        );
    }
    if replaced > 0 {
        println!(
            "  ⚠️  {} 个非法字节被替换为 U+FFFD（语料非 UTF-8）",
            replaced
        );
    }

    // m11：run 目录由 RunDirGuard 统一清理（早退路径同样生效）
    if args.keep_runs {
        let p = run_dir.path().clone();
        run_dir.keep(); // 放弃自动清理
        println!("\n[keep-runs] sorted runs 保留于 {}", p.display());
    }
    Ok(())
}

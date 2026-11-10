// BBPE (Byte-Level BPE) 训练：读取一个 JSONL 文件，使用本项目自研的 BBPE 引擎
// 训练一个 BPE 分词器，并保存为项目自定义格式（JSON）。
//
// 用法示例：
//
//   cargo run --release --bin bbpe_jsonl_bench -- \
//       --input data.jsonl \
//       --output tokenizer.json \
//       --field text \
//       --vocab-size 32000 \
//       --min-frequency 2
//
//   # 常用参数：
//   #   --field <name>        JSON 中存放待分词文本的字段名（默认 text）
//   #   --vocab-size <N>      目标词表大小（默认 32000）
//   #   --min-frequency <N>   合并的最小频次（默认 2，与 TrainerConfig 对齐）
//   #   --max-lines <N>       只读取前 N 行（占位：当前 CorpusReader::JsonlDynamic
//   #                          未内置行数限制，传入仅打印提示）
//   #   --max-token-length <N> 单个 token 的最大长度（占位：本引擎以字节为基本单位，
//   #                          当前训练管线不直接限制单 token 字节数）
//   #   --limit-alphabet <N>  限制初始字母表大小（占位：本引擎固定以 256 字节为初始
//   #                          字母表）
//   #   --show-progress       打印训练进度条（占位：训练器内部始终输出 indicatif）
//   #   --quiet               只打印最终汇总
//   #   --special-token <s>   额外添加的特殊 token（可重复）

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::time::Instant;

use bbpe_rs::{BBPETokenizer, BBPETrainer, CorpusReader, TrainerConfig};
// m13：字段命中统计需要 sonic_rs 的 LazyValue 与 JsonValueTrait（as_str）
use sonic_rs::JsonValueTrait;

#[derive(Debug, Clone)]
struct Cli {
    input: String,
    output: String,
    field: String,
    vocab_size: usize,
    min_frequency: u32,
    /// 占位：当前 CorpusReader::JsonlDynamic 未内置行数限制
    max_lines: Option<usize>,
    /// 占位：本引擎不直接支持
    max_token_length: Option<usize>,
    /// 占位：本引擎以 256 字节为基础字母表
    limit_alphabet: Option<usize>,
    /// 占位：训练器已内置 indicatif 进度条
    show_progress: bool,
    quiet: bool,
    special_tokens: Vec<String>,
}

impl Cli {
    fn parse() -> Result<Self, String> {
        let mut input: Option<String> = None;
        let mut output: Option<String> = None;
        let mut field = "text".to_string();
        let mut vocab_size: usize = 32000;
        let mut min_frequency: u32 = 2;
        let mut max_lines: Option<usize> = None;
        let mut max_token_length: Option<usize> = None;
        let mut limit_alphabet: Option<usize> = None;
        let mut show_progress = false;
        let mut quiet = false;
        let mut special_tokens: Vec<String> = Vec::new();

        let mut args = env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                "--input" => input = Some(args.next().ok_or("missing value for --input")?),
                "--output" => output = Some(args.next().ok_or("missing value for --output")?),
                "--field" => field = args.next().ok_or("missing value for --field")?,
                "--vocab-size" => {
                    vocab_size = args
                        .next()
                        .ok_or("missing value for --vocab-size")?
                        .parse()
                        .map_err(|e| format!("invalid --vocab-size: {e}"))?
                }
                "--min-frequency" => {
                    min_frequency = args
                        .next()
                        .ok_or("missing value for --min-frequency")?
                        .parse()
                        .map_err(|e| format!("invalid --min-frequency: {e}"))?
                }
                "--max-lines" => {
                    max_lines = Some(
                        args.next()
                            .ok_or("missing value for --max-lines")?
                            .parse()
                            .map_err(|e| format!("invalid --max-lines: {e}"))?,
                    )
                }
                "--max-token-length" => {
                    max_token_length = Some(
                        args.next()
                            .ok_or("missing value for --max-token-length")?
                            .parse()
                            .map_err(|e| format!("invalid --max-token-length: {e}"))?,
                    )
                }
                "--limit-alphabet" => {
                    limit_alphabet = Some(
                        args.next()
                            .ok_or("missing value for --limit-alphabet")?
                            .parse()
                            .map_err(|e| format!("invalid --limit-alphabet: {e}"))?,
                    )
                }
                "--show-progress" => show_progress = true,
                "--quiet" => quiet = true,
                "--special-token" => {
                    special_tokens.push(args.next().ok_or("missing value for --special-token")?);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }

        let input = input.ok_or("--input is required")?;
        let output = output.ok_or("--output is required")?;

        Ok(Cli {
            input,
            output,
            field,
            vocab_size,
            min_frequency,
            max_lines,
            max_token_length,
            limit_alphabet,
            show_progress,
            quiet,
            special_tokens,
        })
    }
}

fn print_help() {
    eprintln!(
        "bbpe_jsonl_bench - Train a Byte-Level BPE tokenizer from a JSONL file\n\
         \n\
         USAGE:\n  \
             cargo run --release --bin bbpe_jsonl_bench -- \\\n  \
                 --input <file.jsonl> --output <tokenizer.json> [OPTIONS]\n\
         \n\
         OPTIONS:\n  \
             --input <path>            JSONL input file (required)\n  \
             --output <path>           Output tokenizer.json (required)\n  \
             --field <name>            JSON field containing text (default: text)\n  \
             --vocab-size <N>          Target vocabulary size (default: 32000)\n  \
             --min-frequency <N>       Minimum pair frequency to merge (default: 2)\n  \
             --max-lines <N>           Placeholder (engine streams the full file)\n  \
             --max-token-length <N>    Placeholder (engine uses byte-level BPE)\n  \
             --limit-alphabet <N>      Placeholder (engine uses 256-byte base alphabet)\n  \
             --show-progress           Placeholder (trainer already prints a bar)\n  \
             --special-token <s>       Add an extra special token (repeatable)\n  \
             --quiet                   Print only the final summary"
    );
}

// -------- 内存采样（Linux /proc/self/status） --------

#[derive(Debug, Default, Clone, Copy)]
struct MemSample {
    rss_kb: u64,
    vms_kb: u64,
    hwm_kb: u64,
    vmp_peak_kb: u64,
}

fn read_mem() -> Option<MemSample> {
    let mut s = String::new();
    File::open("/proc/self/status")
        .ok()?
        .read_to_string(&mut s)
        .ok()?;
    let mut m = MemSample::default();
    for line in s.lines() {
        let mut it = line.split_whitespace();
        let key = match it.next() {
            Some(k) => k,
            None => continue,
        };
        let val: u64 = match it.next() {
            Some(v) => v.parse().unwrap_or(0),
            None => continue,
        };
        match key {
            "VmRSS:" => m.rss_kb = val,
            "VmSize:" => m.vms_kb = val,
            "VmHWM:" => m.hwm_kb = val,
            "VmPeak:" => m.vmp_peak_kb = val,
            _ => {}
        }
    }
    Some(m)
}

fn fmt_kb(kb: u64) -> String {
    if kb >= 1024 * 1024 {
        format!("{:.2} GB", kb as f64 / 1024.0 / 1024.0)
    } else if kb >= 1024 {
        format!("{:.2} MB", kb as f64 / 1024.0)
    } else {
        format!("{kb} KB")
    }
}

// -------- 训练主流程 --------

fn main() {
    let cli = match Cli::parse() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}\n");
            print_help();
            std::process::exit(2);
        }
    };

    if !cli.quiet {
        eprintln!(
            "Building BBPE trainer (vocab_size={}, min_frequency={}) ...",
            cli.vocab_size, cli.min_frequency
        );
        if let Some(n) = cli.limit_alphabet {
            eprintln!(
                "note: --limit-alphabet {n} is a no-op for the in-house BBPE engine \
                 (it always starts from the 256-byte alphabet)."
            );
        }
        if let Some(n) = cli.max_token_length {
            eprintln!(
                "note: --max-token-length {n} is a no-op for the in-house BBPE engine \
                 (merge depth is controlled by --vocab-size)."
            );
        }
        if let Some(n) = cli.max_lines {
            eprintln!(
                "note: --max-lines {n} is currently a no-op for the in-house BBPE engine \
                 (CorpusReader::JsonlDynamic streams the full file). Use a head utility to slice."
            );
        }
        if cli.show_progress {
            eprintln!("note: --show-progress is a no-op (trainer always shows a bar).");
        }
    }

    // 构造 TrainerConfig；CLI 的 min_frequency 与 special_tokens 优先级最高
    let mut trainer_config = TrainerConfig::new(cli.vocab_size);
    trainer_config.min_frequency = cli.min_frequency;
    if !cli.special_tokens.is_empty() {
        trainer_config.special_tokens = cli.special_tokens.clone();
    }

    let mut trainer = match BBPETrainer::new(trainer_config, None) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to build trainer: {e}");
            std::process::exit(1);
        }
    };

    if !cli.quiet {
        eprintln!("Reading {} ...", cli.input);
    }

    // 构造 schema_rules：优先字段 + 多个回退字段，全部标记为 "str"。
    // CorpusReader::JsonlDynamic 会按规则顺序逐个尝试提取，直到拿到非空字符串。
    let mut field_candidates: Vec<String> = vec![cli.field.clone()];
    for fb in ["content", "prompt", "input", "sentence", "raw"] {
        if !field_candidates.iter().any(|f| f == fb) {
            field_candidates.push(fb.to_string());
        }
    }
    let schema_rules: Vec<(Vec<String>, String)> = field_candidates
        .iter()
        .map(|f| (vec![f.clone()], "str".to_string()))
        .collect();

    // m13：字段回退链会把"缺 text 的行"静默改抽 content/prompt/...，
    // 而 summary 只显示 `cli.field` ⟹ 用户以为全程用的 text，实际混了好几个字段。
    // 这里按读取器的语义（首个命中的非空字段胜出）统计各字段实际命中数并打印。
    let field_hits = count_field_hits(&cli.input, &field_candidates);

    let corpus_reader = CorpusReader::JsonlDynamic {
        filepath: cli.input.clone(),
        schema_rules,
    };

    let mem_start = read_mem();
    let t_start = Instant::now();

    if let Err(e) = trainer.train(&corpus_reader) {
        eprintln!("training failed: {e}");
        std::process::exit(1);
    }

    let elapsed = t_start.elapsed();
    let mem_end = read_mem();
    let mem_peak = mem_end.and_then(|_| read_peak_rss_kb());

    if let Err(e) = trainer.save(&cli.output) {
        eprintln!("failed to save tokenizer: {e}");
        std::process::exit(1);
    }

    // 顺手做一次冒烟校验（不计入耗时/内存统计），保证 save/load 闭环可用
    if !cli.quiet {
        match BBPETokenizer::load(&cli.output) {
            Ok(tok) => {
                let sample = "BBPE 训练完成，欢迎使用 Rust 极速分词器!";
                let ids = tok.encode_ordinary(sample);
                match tok.decode(&ids) {
                    Ok(decoded) => eprintln!(
                        "smoke test: ids.len={}, decode ok ({} bytes)",
                        ids.len(),
                        decoded.len()
                    ),
                    Err(e) => eprintln!("smoke test decode failed: {e}"),
                }
            }
            Err(e) => eprintln!("smoke test: failed to reload tokenizer: {e}"),
        }
    }

    // m13：把"实际用了哪些字段"讲清楚（回退链此前是完全静默的）
    let (hits, missed) = field_hits;
    let total_hit: u64 = hits.iter().sum();
    println!("-------------------------------------------------------");
    println!("字段命中统计（按回退顺序，首个非空字段胜出）：");
    for (i, field) in field_candidates.iter().enumerate() {
        let n = hits[i];
        let pct = if total_hit + missed > 0 {
            n as f64 / (total_hit + missed) as f64 * 100.0
        } else {
            0.0
        };
        let mark = if field == &cli.field {
            " ← --field"
        } else {
            "（回退）"
        };
        println!("  {:<16} {:>10} 行 ({:>5.2}%){}", field, n, pct, mark);
    }
    if missed > 0 {
        println!(
            "  {:<16} {:>10} 行 —— 所有候选字段均缺失，该行未贡献语料",
            "<未命中>", missed
        );
    }
    if total_hit > 0 && hits[0] < total_hit {
        println!(
            "  ⚠️  有 {} 行并未使用 --field 指定的 {:?}，而是落到了回退字段上；\
             若这不是预期，请检查数据字段名或调整回退链",
            total_hit - hits[0],
            cli.field
        );
    }
    println!("-------------------------------------------------------");

    print_summary(
        &cli,
        trainer.state.vocab.len(),
        elapsed.as_secs_f64(),
        mem_start,
        mem_end,
        mem_peak,
    );
}

/// m13 / R4-14：统计"各候选字段实际命中了多少行"。
///
/// 语义与 `CorpusReader::JsonlDynamic` **完全一致**：逐规则尝试，**全部**命中
/// 字段都会贡献语料（动态 reader 不做"首个命中即停"）。原实现有两个偏差：
///   ① `read_to_end` 整文件读入内存 —— 200G 语料会在训练前就尝试分配等量内存，
///      直接污染 VmHWM 统计甚至 OOM；改为 `read_until` 流式逐行；
///   ② 首个字段命中后 `break` —— 与训练口径（全部命中字段都提取）不符，
///      "字段来源报告"因此失真。改为逐字段累计。
/// 返回的计数与 `candidates` 一一对应；另返回"整行都没命中"的行数，
/// 让"字段回退链静默混字段"这件事变成可见的数字。
fn count_field_hits(path: &str, candidates: &[String]) -> (Vec<u64>, u64) {
    let mut hits = vec![0u64; candidates.len()];
    let mut missed = 0u64;

    let Ok(file) = File::open(path) else {
        return (hits, missed);
    };
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(_) => {}
            // R4-14：IO 错误时停止统计（与 reader.rs 非严格模式同口径：告警后终止）。
            // 注意：统计只是诊断信息，训练本身走 CorpusReader（严格模式）另行失败。
            Err(e) => {
                eprintln!("⚠️  bbpe_jsonl_bench: 字段统计读取 IO 错误，统计提前终止: {e}");
                break;
            }
        }
        let line = buf.strip_suffix(b"\n").unwrap_or(&buf);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let mut any = false;
        for (i, field) in candidates.iter().enumerate() {
            if let Ok(val) = sonic_rs::get_from_slice(line, &[field.as_str()]) {
                let s = val.as_str().unwrap_or("");
                if !s.is_empty() {
                    hits[i] += 1;
                    any = true;
                }
            }
        }
        if !any {
            missed += 1;
        }
    }
    (hits, missed)
}

fn read_peak_rss_kb() -> Option<u64> {
    let mut s = String::new();
    File::open("/proc/self/status")
        .ok()?
        .read_to_string(&mut s)
        .ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let v: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0);
            return Some(v);
        }
    }
    None
}

fn print_summary(
    cli: &Cli,
    vocab_size_after: usize,
    secs: f64,
    mem_start: Option<MemSample>,
    mem_end: Option<MemSample>,
    peak_rss: Option<u64>,
) {
    println!("================ BBPE training summary ================");
    println!("input file          : {}", cli.input);
    println!("output tokenizer    : {}", cli.output);
    println!("field               : {}", cli.field);
    println!("vocab_size          : {}", cli.vocab_size);
    println!("min_frequency       : {}", cli.min_frequency);
    println!("special_tokens      : {}", cli.special_tokens.len());
    if let Some(n) = cli.max_lines {
        println!("max_lines (cap)     : {n} (no-op for engine)");
    }
    println!("-------------------------------------------------------");
    println!("final vocab entries : {vocab_size_after}");
    println!("elapsed (s)         : {secs:.4}");
    println!("-------------------------------------------------------");
    if let Some(s) = mem_start {
        println!(
            "mem @ start  RSS={}, VMS={}",
            fmt_kb(s.rss_kb),
            fmt_kb(s.vms_kb)
        );
    }
    if let Some(e) = mem_end {
        println!(
            "mem @ end    RSS={}, VMS={}",
            fmt_kb(e.rss_kb),
            fmt_kb(e.vms_kb)
        );
    }
    match peak_rss {
        Some(kb) => println!("peak RSS (VmHWM)    : {}", fmt_kb(kb)),
        None => println!("peak RSS            : <unavailable>"),
    }
    println!("=======================================================");
}

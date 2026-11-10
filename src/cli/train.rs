// src/cli/train.rs
//
// 统一训练入口（三支柱 · 支柱一）。**一个命令覆盖两种语料规模**：
//
//   bbpe train --input full.jsonl --vocab 100256           # 不采样 ⟹ 全量精确训练
//   bbpe train --input big.jsonl  --vocab 150256 \
//       --sample-rate 0.20 --heldout-rate 0.005 --heldout-out heldout.jsonl
//                                                          # ⟹ 流式采样 + 训练
//
// 相比旧的 `sample_corpus | train_vocab` 两步走，这里解决三个问题：
//
//   1. **出身信息是事实而非转录**：旧流程要求用户把 sample_corpus 的参数手工抄进
//      `train_vocab --sampling-rate/--seed`，抄错则模型 meta 与实际不符（G1 可追溯失效）。
//      现在采样与训练在同一进程内完成，`meta.sampling` 由程序据实写入。
//   2. **采样不再是"另一条路"**：只是多了几个参数，语料规模这个连续量不该让用户
//      在两个产品之间做选择。
//   3. **OOM 预检代替崩溃**：按 `PEAK_MEM_FACTOR` 估算峰值内存，超预算时给出
//      可直接复制的降采样命令，而不是让进程被 OOM killer 干掉。
//
// 采样语义完全复用 `sampling::sample_lines`（互斥性 / 嵌套性 / held-out 与 rate 无关
// 三条保证由 tests/track_a_test.rs 常驻守护），本文件只做编排与上报。

use super::util::{
    CliResult, PEAK_MEM_FACTOR, TempFileGuard, available_ram_bytes, ensure_parent,
    exceeds_memory_budget, human_bytes, peak_rss_gb, same_path, temp_path,
};
use crate::reader::{CorpusReader, CorpusStats};
use crate::sampling::SampleParams;
use crate::tokenizer::{BpeState, SamplingMeta};
use crate::{BBPETrainer, TrainerConfig};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

struct Args {
    input: String,
    output: String,
    format: String,
    /// 总词表大小（= 256 + merges 数），与其它工具同口径
    vocab: usize,
    min_frequency: u32,
    sample_rate: Option<f64>,
    sample_seed: u64,
    heldout_rate: f64,
    heldout_seed: u64,
    heldout_out: Option<String>,
    /// 保留采样语料（默认写临时文件并在结束后删除）
    sample_out: Option<String>,
    max_bytes: u64,
    /// 自定义预分词正则（默认用 DEFAULT_PATTERN）。用于"同正则对照"实验：
    /// 传入 GPT-2 的 pattern 即可与 HuggingFace tokenizers 做逐条对齐比较。
    pattern: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            input: String::new(),
            output: String::from("output/vocab.json"),
            format: String::from("jsonl"),
            vocab: 100_256,
            min_frequency: 2,
            sample_rate: None,
            sample_seed: 42,
            heldout_rate: 0.0,
            heldout_seed: 43,
            heldout_out: None,
            sample_out: None,
            max_bytes: u64::MAX,
            pattern: None,
        }
    }
}

fn parse(args: Vec<String>) -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let mut v = |flag: &str| it.next().ok_or(format!("{flag} 需要参数"));
        match k.as_str() {
            "--input" => a.input = v("--input")?,
            "--output" => a.output = v("--output")?,
            "--format" => a.format = v("--format")?,
            "--pattern" => a.pattern = Some(v("--pattern")?),
            "--vocab" => a.vocab = v("--vocab")?.parse().map_err(|e| format!("--vocab: {e}"))?,
            "--min-frequency" => {
                a.min_frequency = v("--min-frequency")?
                    .parse()
                    .map_err(|e| format!("--min-frequency: {e}"))?
            }
            "--sample-rate" => {
                a.sample_rate = Some(
                    v("--sample-rate")?
                        .parse()
                        .map_err(|e| format!("--sample-rate: {e}"))?,
                )
            }
            "--sample-seed" => {
                a.sample_seed = v("--sample-seed")?
                    .parse()
                    .map_err(|e| format!("--sample-seed: {e}"))?
            }
            // 旧名兼容：v7 时代的 --sampling-rate 只写 meta、不真采样，是历史坑；
            // 现在同名参数会**真的采样**，出身信息因而才是事实（不再是人工转录）。
            "--sampling-rate" => {
                eprintln!("⚠️  --sampling-rate 已更名为 --sample-rate（且现在会真的采样）");
                a.sample_rate = Some(
                    v("--sampling-rate")?
                        .parse()
                        .map_err(|e| format!("--sampling-rate: {e}"))?,
                )
            }
            "--sampling-seed" => {
                eprintln!("⚠️  --sampling-seed 已更名为 --sample-seed");
                a.sample_seed = v("--sampling-seed")?
                    .parse()
                    .map_err(|e| format!("--sampling-seed: {e}"))?
            }
            "--heldout-rate" => {
                a.heldout_rate = v("--heldout-rate")?
                    .parse()
                    .map_err(|e| format!("--heldout-rate: {e}"))?
            }
            "--heldout-seed" => {
                a.heldout_seed = v("--heldout-seed")?
                    .parse()
                    .map_err(|e| format!("--heldout-seed: {e}"))?
            }
            "--heldout-out" => a.heldout_out = Some(v("--heldout-out")?),
            "--sample-out" => a.sample_out = Some(v("--sample-out")?),
            "--max-bytes" => {
                a.max_bytes = v("--max-bytes")?
                    .parse()
                    .map_err(|e| format!("--max-bytes: {e}"))?
            }
            other => return Err(format!("未知参数: {other}")),
        }
    }
    if a.input.is_empty() {
        return Err("必须提供 --input".into());
    }
    if a.vocab <= 256 {
        return Err("--vocab 是总词表大小（256 基础字节 + merges 数），必须 > 256".into());
    }
    if let Some(r) = a.sample_rate {
        if !(0.0..=1.0).contains(&r) {
            return Err(format!("--sample-rate 必须在 [0,1]，实际 {r}"));
        }
    }
    if !(0.0..=1.0).contains(&a.heldout_rate) {
        return Err(format!(
            "--heldout-rate 必须在 [0,1]，实际 {}",
            a.heldout_rate
        ));
    }
    if a.heldout_rate > 0.0 && a.heldout_out.is_none() {
        return Err(
            "--heldout-rate > 0 时必须同时提供 --heldout-out（否则命中行会被静默丢弃）".into(),
        );
    }
    if a.heldout_rate == 0.0 && a.heldout_out.is_some() {
        return Err(
            "提供了 --heldout-out 但 --heldout-rate = 0：held-out 文件将为空，请补 --heldout-rate"
                .into(),
        );
    }
    // R4-22：输入/输出别名防护。采样边读边写同一文件会让输出污染输入、
    // 输入被自身截断（File::create 先截断为 0）；--sample-out/--heldout-out
    // 与 --input 别名时同理。与 cli/sample.rs 同口径。
    if let Some(s) = &a.sample_out {
        if same_path(&a.input, s) {
            return Err(format!(
                "--input 与 --sample-out 指向同一个文件（{}）：采样输出会覆盖原始语料",
                a.input
            ));
        }
        if same_path(s, &a.output) {
            return Err(format!(
                "--sample-out 与 --output 指向同一个文件（{s}）：采样语料与模型文件不能共路径"
            ));
        }
    }
    if let Some(h) = &a.heldout_out {
        if same_path(&a.input, h) {
            return Err(format!(
                "--input 与 --heldout-out 指向同一个文件（{}）：held-out 输出会覆盖原始语料",
                a.input
            ));
        }
        if let Some(s) = &a.sample_out {
            if same_path(h, s) {
                return Err(format!(
                    "--heldout-out 与 --sample-out 指向同一个文件（{h}）：两者内容不同，共路径会互相覆盖"
                ));
            }
        }
    }
    Ok(a)
}

pub fn run(args: Vec<String>) -> CliResult {
    super::util::init_tracing_once();
    let a = parse(args)?;
    ensure_parent(&a.output)?;

    let sampling = a.sample_rate.is_some() || a.heldout_rate > 0.0;
    let rate = a.sample_rate.unwrap_or(1.0);
    let corpus_bytes = std::fs::metadata(&a.input).map(|m| m.len()).unwrap_or(0);

    println!("== bbpe train（统一训练入口） ==");
    println!(
        "  语料: {} [{}] | {}{}",
        a.input,
        a.format,
        human_bytes(corpus_bytes),
        if sampling {
            format!(" | 采样率 {:.2}%", rate * 100.0)
        } else {
            " | 全量精确训练（不采样）".to_string()
        }
    );
    println!(
        "  目标总词表 {}（= 256 基础 + {} merges）| min_frequency {}",
        a.vocab,
        a.vocab - 256,
        a.min_frequency
    );

    // R6-1：SIGINT 令牌提前到采样之前 —— 采样预读同样是分钟级的长任务，
    // 必须能在这一阶段就停下（sample_lines_cancel 每 4096 行检查一次）。
    let cancel = crate::cli::util::sigint_cancel_token();

    // n2：临时采样语料交给 RAII 守卫 —— 训练失败 / 参数非法 / Ctrl+C 取消
    // 等任何提前返回都会自动清理。守卫必须在**文件创建之后立即**建立：
    // 若等采样成功再建，采样中途被取消时文件已存在却无人清理
    // （R6-1 发布验收在 CLI 采样阶段踩到过这个泄漏）。
    let mut _sample_guard: Option<TempFileGuard> = None;

    // ---------- 阶段 1：采样（可选）----------
    // 采样在**物理行**层面进行（与 sample_corpus 完全一致的语义），
    // 训练仍在采样产出的文件上跑 ⟹ 与"两步走"逐字节等价，只是不必落两步命令。
    let train_corpus = if sampling {
        let path = match &a.sample_out {
            Some(p) => {
                ensure_parent(p)?;
                std::path::PathBuf::from(p)
            }
            None => temp_path("bbpe_train_sample", ".jsonl"),
        };
        let is_temp = a.sample_out.is_none();

        let fin = File::open(&a.input)?;
        let mut reader = BufReader::with_capacity(1 << 20, fin);
        let mut train_w = BufWriter::with_capacity(1 << 20, File::create(&path)?);
        // 文件一旦创建就交给守卫：采样中途的取消 / IO 错误都必须清理它
        if is_temp {
            _sample_guard = Some(TempFileGuard::new(path.clone()));
        }
        let mut held_w = match &a.heldout_out {
            Some(p) => {
                ensure_parent(p)?;
                Some(BufWriter::with_capacity(1 << 20, File::create(p)?))
            }
            None => None,
        };

        let s = crate::sampling::sample_lines_cancel(
            &mut reader,
            &mut train_w,
            held_w.as_mut(),
            SampleParams {
                rate,
                seed: a.sample_seed,
                heldout_rate: a.heldout_rate,
                heldout_seed: a.heldout_seed,
                max_bytes: a.max_bytes,
            },
            Some(cancel.as_ref()),
        )
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::Interrupted {
                "采样已被 Ctrl+C 取消；临时文件已清理，原始语料未做任何改动".to_string()
            } else {
                e.to_string()
            }
        })?;
        train_w.flush()?;
        drop(train_w);
        if let Some(w) = held_w.as_mut() {
            w.flush()?;
        }
        drop(held_w);

        println!(
            "  采样: 总 {} 行 ⟹ 训练 {} 行（{}，实际率 {:.2}%，seed={}）",
            s.total_lines,
            s.train_lines,
            human_bytes(s.train_bytes),
            s.train_bytes as f64 / s.total_bytes.max(1) as f64 * 100.0,
            a.sample_seed
        );
        if let Some(p) = &a.heldout_out {
            println!(
                "  held-out: {} | {} 行（{}，与训练集互斥，seed={}）",
                p,
                s.held_lines,
                human_bytes(s.held_bytes),
                a.heldout_seed
            );
        } else {
            println!("  held-out: 未启用（--heldout-rate = 0）");
        }
        if is_temp {
            println!("  采样语料: 临时文件（训练结束后自动删除；加 --sample-out 可保留）");
        }
        path.to_string_lossy().to_string()
    } else {
        a.input.clone()
    };

    // 守卫已在文件创建时绑定（见阶段 1 内），这里只需确认它活到函数结束。

    // ---------- 阶段 2：内存预检 ----------
    // 与其让 OOM killer 在半小时后干掉进程，不如在开工前把话说清楚。
    //
    // R4-1（P0）：预检失败**绝不能删除任何输入文件**。原实现在返回错误前
    // `remove_file(&train_corpus)`：不采样时 train_corpus 就是**用户原始语料**，
    // 于是"语料装不进内存"这一条错误提示的代价是原始数据被删；
    // 即使用 --sample-out 显式保留样本，也会因 is_temp=false、守卫为空同样被删。
    // 临时采样文件的清理已由 _sample_guard（RAII）负责，此处无需也不得删文件。
    let train_bytes = std::fs::metadata(&train_corpus)
        .map(|m| m.len())
        .unwrap_or(0);
    let estimate = train_bytes as f64 * PEAK_MEM_FACTOR;
    // R4-1：判定抽为纯函数（阈值可注入，单测确定性覆盖边界）；
    // 行为不变：读不到 MemAvailable 时不拦。
    let avail_opt = available_ram_bytes();
    if exceeds_memory_budget(train_bytes, avail_opt) {
        let avail = avail_opt.unwrap_or(0);
        let suggest =
            ((avail as f64 * 0.8 / PEAK_MEM_FACTOR) / corpus_bytes.max(1) as f64).clamp(0.01, 1.0);
        return Err(format!(
            "语料装不进内存：训练语料 {} × {PEAK_MEM_FACTOR} ≈ {}，可用内存约 {}。\n\
             原始输入文件**未做任何改动**；建议加降采样重跑（其余参数不变）：\n\
               bbpe train --input {} --vocab {} \\\n\
                   --sample-rate {:.3} --sample-seed {} --heldout-rate 0.005 \\\n\
                   --heldout-out heldout.jsonl",
            human_bytes(train_bytes),
            human_bytes(estimate as u64),
            human_bytes(avail),
            a.input,
            a.vocab,
            suggest,
            a.sample_seed,
        )
        .into());
    }

    // ---------- 阶段 3：训练 ----------
    let reader = match a.format.as_str() {
        "jsonl" => CorpusReader::JsonlStaticText {
            filepath: train_corpus.clone(),
        },
        "text" => CorpusReader::Text {
            filepath: train_corpus.clone(),
        },
        // n2：清理已由 _sample_guard 负责，此处无需显式删除
        other => return Err(format!("未知 --format: {other}（支持 jsonl | text）").into()),
    };

    let mut config = TrainerConfig::new(a.vocab);
    config.min_frequency = a.min_frequency;
    if let Some(pat) = &a.pattern {
        // 只改正则，不做其它语义变更；pattern 会在 trainer 构造时被校验
        // （空 pattern / 未覆盖间隙会被拒绝），校验失败即报错退出。
        config.pre_tokenizer_pattern = pat.clone();
    }

    let t0 = Instant::now();
    let mut trainer = BBPETrainer::new(config, None)?;
    let stats = Arc::new(CorpusStats::default());
    // R6-1：SIGINT 变成取消令牌后，Err(Cancelled) 会正常展开调用栈，
    // _sample_guard 的 Drop 得以清理临时采样文件；原始语料始终不改动。
    // 令牌在采样阶段之前创建，这里复用同一个实例。
    match trainer.train_with_cancel(&reader, Some(stats.clone()), Some(cancel.as_ref())) {
        Ok(()) => {}
        Err(crate::error::BbpeError::Cancelled { merges_done, .. }) => {
            return Err(format!(
                "训练已被 Ctrl+C 取消（已完成 {merges_done} 次合并）；\
                 临时采样文件已清理，原始语料未做任何改动"
            )
            .into());
        }
        Err(e) => return Err(e.into()),
    }

    // 出身信息（G1）：采样参数**由程序据实写入**，不再依赖人工转录。
    // 未采样时 meta.sampling 保持缺省 ⟹ "全量训练"与"采样训练"在模型上一眼可辨。
    //
    // A5：profile 一并在此写入（此前只有 Python 绑定写、CLI 不写、inspect 不展示
    // ⟹ 三方不齐，该字段事实上没有契约价值）。CLI 与 Python 绑定都是核心剖面，
    // 故统一标 "v2"（ADR-1：剖面仅为元信息，不代表格式差异）。
    trainer.state.meta_mut().profile = Some("v2".to_string());
    if sampling {
        trainer.state.meta_mut().sampling = Some(SamplingMeta {
            rate,
            seed: a.sample_seed,
            heldout_rate: a.heldout_rate,
            heldout_seed: a.heldout_seed,
        });
    }
    trainer.save(&a.output)?;
    let secs = t0.elapsed().as_secs_f64();

    // 输入侧静默行为必须可见（规范 §6.6 / ADR-5：行为不变，但上报）
    let (total_lines, total_bytes, skipped_lines, replacement_chars) = stats.snapshot();
    println!(
        "  训练语料: {} 行 | {} | skipped_lines {} | replacement_chars {}",
        total_lines,
        human_bytes(total_bytes),
        skipped_lines,
        replacement_chars
    );
    if skipped_lines > 0 {
        println!(
            "  ⚠️  {} 行被静默跳过（JSONL 坏行 / 缺字段），占比 {:.4}%",
            skipped_lines,
            skipped_lines as f64 / total_lines.max(1) as f64 * 100.0
        );
    }
    if replacement_chars > 0 {
        println!(
            "  ⚠️  {} 个非法字节被替换为 U+FFFD（语料非 UTF-8）",
            replacement_chars
        );
    }

    // 回读模型：既校验 save/load 往返，也拿到真实 merges 数
    let state = BpeState::load(&a.output)?;
    let actual_merges = state.merges.len();
    let target_merges = a.vocab - 256;

    println!();
    println!("  耗时 {:.1}s | 峰值内存 {:.2} GB", secs, peak_rss_gb());
    println!(
        "  词表 {} 项（= 256 基础 + {} merges）",
        state.vocab.len(),
        actual_merges
    );
    println!();
    println!("== 推论 7 判据（vocab_size 是否先到） ==");
    // B2：判据必须读**真值**，不能反推。
    //
    // 旧实现用 `actual_merges >= target_merges` 反推终止原因，而"提前终止"在
    // M2 把 `no_more_pairs` 从 `min_frequency_terminated` 里拆出来之后已有
    // **三种**成因，反推只能得到其中一种。实测（300 行小语料 / --vocab 300）：
    // 模型自述 `end_reason=no_more_pairs`，CLI 却打印"min_frequency=2 才是终止
    // 原因"，并给出**方向错误**的处置建议（让下调 K 或增大样本 —— 而真实原因
    // 与 m 无关，下调 m / K 都无济于事，需要更丰富多样的语料）。
    // 这与 A3 是同一类问题：修了一处、漏了兄弟，且两处都在描述同一件事。
    let end_reason = state
        .meta
        .as_ref()
        .and_then(|m| m.train.as_ref())
        .map(|t| t.end_reason.as_str());
    match end_reason {
        Some("vocab_size_reached") => {
            println!(
                "  ✅ 成立：实际 merges {} ≥ 目标 {} ⟹ vocab_size 是终止原因，\
                 第 K 次合并时仍有 pair 频次 ≥ min_frequency={}",
                actual_merges, target_merges, a.min_frequency
            );
            println!("     ⟹ 推论 7 生效：m=2 与 m=100 输出 bit-identical，严格/近似模式合一。");
        }
        Some("min_frequency_terminated") => {
            println!(
                "  ⚠️  失效：实际 merges {} < 目标 {} ⟹ min_frequency={} 才是终止原因",
                actual_merges, target_merges, a.min_frequency
            );
            println!(
                "     ⟹ 该语料规模不足以支撑总词表 {}；m 的选择开始影响输出，需下调 K 或增大样本。",
                a.vocab
            );
        }
        Some("no_more_pairs") => {
            println!(
                "  ⚠️  失效：实际 merges {} < 目标 {} ⟹ 语料的可合并空间已耗尽（no_more_pairs）",
                actual_merges, target_merges
            );
            println!(
                "     ⟹ 与 min_frequency={} **无关**：下调 m 或 K 都不会增加 merges，\
                 需要更丰富 / 更多样的语料 —— 当前语料的 pair 多样性已用完。",
                a.min_frequency
            );
        }
        other => {
            println!(
                "  ⚠️  模型未记录可识别的终止原因（{:?}）；按 merges 数反推：实际 {} / 目标 {}",
                other, actual_merges, target_merges
            );
            println!("     ⟹ 请用 `bbpe inspect <model>` 复核 meta.train 段后人工判读。");
        }
    }
    println!();
    println!("  模型已保存: {}", a.output);

    if let Some(p) = &a.heldout_out {
        println!();
        println!("  下一步（用 held-out 选 K，单位与 --vocab 一致，都是总词表）：");
        println!(
            "    bbpe sweep --model {} --held-out {} --vocabs 8448,16640,33024,65792,100256",
            a.output, p
        );
    }

    // n2：临时采样语料的清理交给 _sample_guard（作用域结束时自动执行）
    Ok(())
}

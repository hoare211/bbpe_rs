// src/cli/inspect.rs
//
// 模型体检与迁移（`bbpe inspect`）。模型契约工具，**与发布形态无关**。
//
// 三种模式：
//   1) bbpe inspect <model> [--allow-legacy-pattern]
//        打印出身与规格摘要，并执行 V1–V5 自洽性校验
//   2) bbpe inspect <model> --validate-only [--allow-legacy-pattern]
//        只校验，退出码 0 = 通过 / 1 = 失败（供 CI 用）
//   3) bbpe inspect --migrate <in> <out> --pattern <regex>
//        v0 → v1：补写 pre_tokenizer_pattern 与 format_version
//
// ⚠️ migrate 的 pattern **必须由用户显式指定**——猜 pattern 只是把 Q2
//    "静默 fallback" 这个错误换个地方犯，故缺 --pattern 直接报错。

use super::util::{CliResult, ensure_parent};
// A4：不再导入 MODEL_FORMAT —— inspect 必须显示**文件里真实声明的** format，
// 而不是这个常量，否则无论文件内容如何都会打印出 `bbpe-rs-model`。
use crate::tokenizer::{BpeState, FORMAT_VERSION, LoadOptions};

fn opt<T: std::fmt::Display>(v: &Option<T>) -> String {
    match v {
        Some(v) => v.to_string(),
        None => "未记录".to_string(),
    }
}

/// pattern 通常很长（多行 (?x) 写法），只显示单行摘要
fn brief_pattern(p: &str) -> String {
    let one_line: String = p.chars().take(80).collect();
    let collapsed = one_line.replace('\n', " ");
    if p.chars().count() > 80 {
        format!("{collapsed} …（共 {} 字符）", p.chars().count())
    } else {
        collapsed
    }
}

fn print_summary(path: &str, st: &BpeState) {
    println!("== bbpe inspect（模型契约 v1） ==");
    println!("  文件: {path}");
    // A4：显示**文件里真实声明的** format / format_version，而不是常量。
    // 原实现打印的是 `MODEL_FORMAT` 常量，于是无论文件是否声明 format，
    // 输出都是 `bbpe-rs-model` —— v0 旧模型被展示成"格式正确"，
    // 契约诊断工具在契约字段上误报，比不诊断更糟。
    match (&st.format, &st.format_version) {
        (Some(f), Some(v)) => {
            println!("  format: {f} | format_version: {v}（本程序支持 {FORMAT_VERSION}）")
        }
        (Some(f), None) => println!(
            "  format: {f} | format_version: **缺失**（v1 起两者必须同时存在 —— 半新半旧，疑为损坏或手工编辑）"
        ),
        (None, Some(v)) => println!(
            "  format: **缺失** | format_version: {v}（v1 起两者必须同时存在 —— 半新半旧，疑为损坏或手工编辑）"
        ),
        (None, None) => println!(
            "  format: **缺失** | format_version: **缺失**（v0 旧格式，建议用 --migrate 迁移）"
        ),
    }
    println!(
        "  词表 {} 项 | merges {} | special tokens {}",
        st.vocab.len(),
        st.merges.len(),
        st.special_tokens.len()
    );
    println!(
        "  pre_tokenizer_pattern: {}",
        brief_pattern(&st.pre_tokenizer_pattern)
    );

    let mut specials: Vec<_> = st.special_tokens.iter().collect();
    specials.sort_by_key(|(_, id)| **id);
    if specials.is_empty() {
        println!("  special tokens: 无");
    } else {
        println!("  special tokens: {} 个", specials.len());
        for (text, id) in specials.iter().take(16) {
            println!("      {id} -> {text}");
        }
        if specials.len() > 16 {
            println!("      …（其余 {} 个略）", specials.len() - 16);
        }
    }

    println!("  meta:");
    match &st.meta {
        None => println!("      （无 meta 块，可能是 --no-meta 产出或 v0 旧模型）"),
        Some(m) => {
            println!("      created_at:      {}", opt(&m.created_at));
            println!("      bbpe_rs_version: {}", opt(&m.bbpe_rs_version));
            println!("      git_commit:      {}", opt(&m.git_commit));
            // A5：profile 是 §7.1 的正式字段，此前 Python 绑定会写、CLI 不写、
            // 且 inspect 从不展示 ⟹ 三方不齐使该字段事实上没有契约价值。
            println!("      profile:         {}", opt(&m.profile));
            match &m.corpus {
                Some(c) => println!(
                    "      corpus:          reader={} lines={} bytes={} skipped={} replacement={}",
                    c.reader, c.total_lines, c.total_bytes, c.skipped_lines, c.replacement_chars
                ),
                None => println!("      corpus:          未记录"),
            }
            match &m.sampling {
                Some(s) => println!(
                    "      sampling:        rate={} seed={} heldout_rate={} heldout_seed={}（采样训练）",
                    s.rate, s.seed, s.heldout_rate, s.heldout_seed
                ),
                None => println!("      sampling:        未记录（全量训练，未采样）"),
            }
            match &m.train {
                // m6：截断产出的模型必须显示"从哪个规模来、现在实际多大"，
                // 否则只看 vocab_size 会误判模型规模。
                Some(t) => println!(
                    "      train:           vocab_size={} min_frequency={} actual_merges={} end_reason={} max_chunk_bytes={}{}",
                    t.vocab_size,
                    t.min_frequency,
                    t.actual_merges,
                    t.end_reason,
                    t.max_chunk_bytes,
                    match t.truncated_from {
                        Some(src) =>
                            format!(" [截断自 vocab_size={src}，实际规模以 actual_merges 为准]"),
                        None => String::new(),
                    }
                ),
                None => println!("      train:           未记录"),
            }
            match &m.eval {
                Some(e) => println!(
                    "      eval:            held_out_bytes={} fertility_at_k={} 档",
                    opt(&e.held_out_bytes),
                    e.fertility_at_k.as_ref().map(|m| m.len()).unwrap_or(0)
                ),
                None => println!("      eval:            未记录"),
            }
        }
    }
}

pub fn run(args: Vec<String>) -> CliResult {
    let mut model = String::new();
    let mut mig_in = String::new();
    let mut mig_out = String::new();
    let mut pattern: Option<String> = None;
    let mut validate_only = false;
    let mut allow_legacy = false;

    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        match k.as_str() {
            "--migrate" => {
                mig_in = it.next().ok_or("--migrate 需要 <in>")?;
                mig_out = it.next().ok_or("--migrate 需要 <out>")?;
            }
            "--pattern" => pattern = Some(it.next().ok_or("--pattern 需要参数")?),
            "--validate-only" => validate_only = true,
            "--allow-legacy-pattern" => allow_legacy = true,
            other if other.starts_with('-') => return Err(format!("未知参数: {other}").into()),
            other => model = other.to_string(),
        }
    }

    // ================= 模式 3：迁移 =================
    if !mig_in.is_empty() {
        let pat = pattern
            .clone()
            .ok_or("migrate 必须显式提供 --pattern <训练时使用的正则>：旧模型没有记录它，猜不得")?;

        // 迁移对象本身就是"待修的模型"⟹ 加载时关闭校验，修完再校验并报告
        let mut st = BpeState::load_with_options(
            &mig_in,
            LoadOptions {
                allow_legacy: true,
                validate: false,
            },
        )?;
        st.pre_tokenizer_pattern = pat.clone();
        ensure_parent(&mig_out)?;
        st.save(&mig_out)?;

        println!("== bbpe inspect --migrate（v0 → v1） ==");
        println!(
            "  输入: {mig_in}（原 format_version={}）",
            opt(&st.format_version)
        );
        println!("  输出: {mig_out}（format_version={FORMAT_VERSION}）");
        println!("  写入 pre_tokenizer_pattern: {}", brief_pattern(&pat));
        println!("  meta: 原样保留（若有）");
        match st.validate() {
            Ok(()) => println!("  ✅ 迁移后 V1–V5 自洽性校验通过"),
            Err(e) => {
                println!("  ❌ 迁移后校验仍失败：{e}");
                println!("     （文件已写出，便于人工检查；该模型本身存在契约违规，非迁移所致）");
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    // ================= 模式 1 / 2：查看 / 校验 =================
    if model.is_empty() {
        return Err(
            "用法: bbpe inspect <model> [--validate-only] [--allow-legacy-pattern]\n      \
             bbpe inspect --migrate <in> <out> --pattern <regex>"
                .into(),
        );
    }

    // 诊断工具：即便模型缺 pre_tokenizer_pattern（v0）也先给出摘要（规模/出身），
    // 再报告契约违规并 exit 1 —— 用户应能先"看见"模型再决定怎么处置
    let mut missing_pattern = false;
    let st = if allow_legacy {
        BpeState::load_with_options(
            &model,
            LoadOptions {
                allow_legacy: true,
                validate: false,
            },
        )?
    } else {
        match BpeState::load_with_options(
            &model,
            LoadOptions {
                allow_legacy: false,
                validate: false,
            },
        ) {
            Ok(st) => st,
            Err(crate::BbpeError::MissingPreTokenizerPattern) => {
                missing_pattern = true;
                println!(
                    "⚠️  模型缺少 pre_tokenizer_pattern（v0 旧格式）：\
                     以下信息以 DEFAULT_PATTERN 回退展示，仅供诊断"
                );
                BpeState::load_with_options(
                    &model,
                    LoadOptions {
                        allow_legacy: true,
                        validate: false,
                    },
                )?
            }
            Err(e) => return Err(e.into()),
        }
    };

    if !validate_only {
        print_summary(&model, &st);
    }

    let validation = st.validate();
    match &validation {
        Ok(()) => {
            if validate_only {
                println!("✅ {model}: V1–V5 自洽性校验通过");
            } else {
                println!("  ✅ V1–V5 自洽性校验通过");
            }
        }
        Err(e) => {
            if validate_only {
                println!("❌ {model}: {e}");
            } else {
                println!("  ❌ {e}");
            }
        }
    }
    // 缺 pattern 本身即契约违规（Q2）：validate() 此时跑在回退 pattern 上，
    // 结果不能代表真实契约状态，故单独判定
    if missing_pattern {
        println!(
            "❌ {model}: 缺少 pre_tokenizer_pattern（v0 旧格式）—— \
             请用 --migrate 迁移，或加 --allow-legacy-pattern 显式回退"
        );
    }
    if validation.is_err() || missing_pattern {
        std::process::exit(1);
    }
    Ok(())
}

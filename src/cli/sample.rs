// src/cli/sample.rs
//
// 进阶工具：单独采一次样，供多次训练复用（嵌套性标定 / 采样率对比实验）。
//
// 日常训练**不需要它**：`bbpe train --sample-rate ...` 已把采样内建（见 train.rs）。
// 只有当你需要"同一份样本训多个模型"或"做 rate₁ ⊆ rate₂ 嵌套对照"时才用本命令。

use super::util::{CliResult, ensure_parent, human_bytes, same_path};
use crate::sampling::{SampleParams, sample_lines};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

struct Args {
    input: String,
    train_out: String,
    rate: f64,
    seed: u64,
    heldout_out: Option<String>,
    heldout_rate: f64,
    heldout_seed: u64,
    max_bytes: u64,
}

fn parse(args: Vec<String>) -> Result<Args, String> {
    let mut a = Args {
        input: String::new(),
        train_out: String::new(),
        rate: 0.2,
        seed: 42,
        heldout_out: None,
        heldout_rate: 0.0,
        heldout_seed: 43,
        max_bytes: u64::MAX,
    };
    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let mut v = |flag: &str| it.next().ok_or(format!("{flag} 需要参数"));
        match k.as_str() {
            "--input" => a.input = v("--input")?,
            "--train-out" => a.train_out = v("--train-out")?,
            "--rate" => a.rate = v("--rate")?.parse().map_err(|e| format!("--rate: {e}"))?,
            "--seed" => a.seed = v("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--heldout-out" => a.heldout_out = Some(v("--heldout-out")?),
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
            "--max-bytes" => {
                a.max_bytes = v("--max-bytes")?
                    .parse()
                    .map_err(|e| format!("--max-bytes: {e}"))?
            }
            other => return Err(format!("未知参数: {other}")),
        }
    }
    if a.input.is_empty() || a.train_out.is_empty() {
        return Err("必须提供 --input 与 --train-out".into());
    }
    // R4-22：输入/输出别名会让采样边读边写**同一个文件** ⟹ 读到刚写出的行、
    // 输出被自身截断污染。必须在开写前拒绝（File::create 已把输入截断为 0，
    // 事中发现也救不回原始语料）。
    if same_path(&a.input, &a.train_out) {
        return Err(format!(
            "--input 与 --train-out 指向同一个文件（{}）：采样输出会覆盖原始语料",
            a.input
        ));
    }
    if let Some(h) = &a.heldout_out {
        if same_path(&a.input, h) {
            return Err(format!(
                "--input 与 --heldout-out 指向同一个文件（{}）：held-out 输出会覆盖原始语料",
                a.input
            ));
        }
        if same_path(h, &a.train_out) {
            return Err(format!(
                "--heldout-out 与 --train-out 指向同一个文件（{h}）：两者内容不同，共路径会互相覆盖"
            ));
        }
    }
    if a.heldout_rate > 0.0 && a.heldout_out.is_none() {
        return Err("--heldout-rate > 0 时必须同时提供 --heldout-out".into());
    }
    if !(0.0..=1.0).contains(&a.rate) {
        return Err("--rate 必须在 [0,1]".into());
    }
    if !(0.0..=1.0).contains(&a.heldout_rate) {
        return Err("--heldout-rate 必须在 [0,1]".into());
    }
    Ok(a)
}

pub fn run(args: Vec<String>) -> CliResult {
    let a = parse(args)?;
    ensure_parent(&a.train_out)?;

    let fin = File::open(&a.input)?;
    let mut reader = BufReader::with_capacity(1 << 20, fin);
    let mut train_w = BufWriter::with_capacity(1 << 20, File::create(&a.train_out)?);
    let mut held_w = match &a.heldout_out {
        Some(p) => {
            ensure_parent(p)?;
            Some(BufWriter::with_capacity(1 << 20, File::create(p)?))
        }
        None => None,
    };

    let s = sample_lines(
        &mut reader,
        &mut train_w,
        held_w.as_mut(),
        SampleParams {
            rate: a.rate,
            seed: a.seed,
            heldout_rate: a.heldout_rate,
            heldout_seed: a.heldout_seed,
            max_bytes: a.max_bytes,
        },
    )?;
    train_w.flush()?;
    drop(train_w);
    if let Some(w) = held_w.as_mut() {
        w.flush()?;
    }
    drop(held_w);

    if s.total_bytes >= a.max_bytes {
        println!("[截断] 已达 --max-bytes 上限");
    }

    println!("== bbpe sample（行级采样） ==");
    println!(
        "  输入: {} | 总行 {} | 总字节 {}",
        a.input,
        s.total_lines,
        human_bytes(s.total_bytes)
    );
    println!(
        "  训练集: {} | {} 行 | {}（实际率 {:.2}%，seed={}）",
        a.train_out,
        s.train_lines,
        human_bytes(s.train_bytes),
        s.train_bytes as f64 / s.total_bytes.max(1) as f64 * 100.0,
        a.seed
    );
    if let Some(p) = &a.heldout_out {
        println!(
            "  held-out: {} | {} 行 | {}（率 {:.3}%，seed={}，与训练集互斥）",
            p,
            s.held_lines,
            human_bytes(s.held_bytes),
            a.heldout_rate * 100.0,
            a.heldout_seed
        );
    }
    println!();
    // R4-22：下一步建议不能再带 --sample-rate —— 本命令产出的 train_out
    // **已经**是采样结果，再采样就是对样本二次抽样（rate² 效应），
    // 与"复用这份样本训练"的意图相反。
    println!("  下一步（样本已采集，直接全量精确训练，请勿再次 --sample-rate）：");
    println!(
        "    bbpe train --input {} --vocab 150256 --output vocab.json",
        a.train_out
    );
    Ok(())
}

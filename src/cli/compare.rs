// src/cli/compare.rs
//
// 合并序列对照：回答"采样率 / 语料规模是否足够"这个核心问题。
//
// **本文件只是打印壳**：指标实现（字节空间还原、逐位一致率、集合重合率、
// id 空间对照）在 `crate::compare`，与 Python 绑定 `compare_models` /
// `BBPETokenizer.compare` 共用同一份代码。方法学说明（为什么必须比字节空间、
// 三个指标各自的口径与判据）见 `src/compare.rs` 的模块文档。

use super::util::{CliResult, parse_num_list};
use crate::compare::{DEFAULT_AT, compare_states};
use crate::tokenizer::{BpeState, LoadOptions};

pub fn run(args: Vec<String>) -> CliResult {
    let mut a_path = String::new();
    let mut b_path = String::new();
    let mut at: Vec<usize> = DEFAULT_AT.to_vec();
    let mut allow_legacy = false;

    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        match k.as_str() {
            "--a" => a_path = it.next().ok_or("--a 需要参数")?,
            "--b" => b_path = it.next().ok_or("--b 需要参数")?,
            "--allow-legacy-pattern" => allow_legacy = true,
            "--at" => at = parse_num_list(&it.next().ok_or("--at 需要参数")?, "--at")?,
            other => return Err(format!("未知参数: {other}").into()),
        }
    }
    if a_path.is_empty() || b_path.is_empty() {
        return Err("必须提供 --a 与 --b".into());
    }

    let opts = LoadOptions {
        allow_legacy,
        validate: true,
    };
    let sa = BpeState::load_with_options(&a_path, opts.clone())?;
    let sb = BpeState::load_with_options(&b_path, opts)?;
    let report = compare_states(&sa, &sb, &at);

    println!("== bbpe compare（合并序列对照，字节空间） ==");
    println!("  A: {} | merges {}", a_path, report.merges_a);
    println!("  B: {} | merges {}", b_path, report.merges_b);
    println!();

    // ① 逐位一致：第一个分叉位置（字节空间）
    // `compared == 0` 单独成句：此时"逐位完全一致（前 0 项）"是假阳性 ——
    // 两侧根本没有可比的合并，既不能说一致也不能说分叉。
    if report.compared == 0 {
        println!(
            "  ⚠️  无可比较的合并（A merges {} / B merges {}）：至少一侧为 0，无法判断二者关系",
            report.merges_a, report.merges_b
        );
    } else {
        match (&report.first_divergence, &report.divergence) {
            (None, _) => println!("  ✅ 逐位完全一致（前 {} 项）", report.compared),
            (Some(i), Some(d)) => println!(
                "  ⚠️  首个分叉在第 {} 次合并（1-based 第 {} 次）：\n     A = {} + {}\n     B = {} + {}",
                i,
                i + 1,
                d.a.0,
                d.a.1,
                d.b.0,
                d.b.1,
            ),
            // 报告自洽性由 `compare_states` 保证（二者同生共死）；此分支仅为穷尽 match。
            (Some(i), None) => {
                println!("  ⚠️  首个分叉在第 {} 次合并（1-based 第 {} 次）", i, i + 1)
            }
        }
        match report.positional.ratio {
            Some(ratio) => println!(
                "     前 {} 项逐位一致 {} 项（{:.2}%）",
                report.compared,
                report.positional.matched,
                ratio * 100.0
            ),
            None => println!(
                "     前 {} 项逐位一致 {} 项（无可比较项）",
                report.compared, report.positional.matched
            ),
        }
    }
    println!();

    // ② 分段指标
    println!(
        "{:>10} {:>12} {:>14} {:>16}",
        "N(前N项)", "逐位一致", "集合重合(字节)", "id空间重合(对照)"
    );
    for seg in &report.at {
        println!(
            "{:>10} {:>11.2}% {:>13.2}% {:>15.2}%",
            seg.k, seg.positional_ratio, seg.set_overlap_ratio, seg.id_space_ratio
        );
    }
    println!();
    println!("  读法：① 『逐位一致』= 第 i 次合并的字节串对严格相同（最严格口径）。");
    println!(
        "        ② 『集合重合(字节)』= 同一批 merge 是否都被选中（不论次序）⟹ 词表内容是否等价。"
    );
    println!(
        "        ③ 『id空间重合』仅供对照：id 由合并次序定义，分叉后必然整体错位，不可作结论依据。"
    );
    println!("        ④ 判据：分叉点越靠后越好；若集合重合远高于逐位一致 ⟹ 只是尾部 pair 频次接近");
    println!("           导致的换位（对下游影响有限），而非系统性偏差。");
    Ok(())
}

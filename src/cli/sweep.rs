// src/cli/sweep.rs
//
// 词表规模扫描：一次训练，评估全部候选规模 —— **用数据选 vocab_size，而不是拍脑袋**。
//
// 原理：BPE 的贪心前缀性质 ⟹ **前 k 次合并 == 总词表 256+k 训练所得的合并序列**（严格成立）。
//       而本项目 `new_id = 256 + i` 把合并次序编码进了 id ⟹ 直接按 id 截断即得前 k 次合并的词表。
//       ⟹ 只需以 K_max 训一次，即可离线评估 K ≤ K_max 的全部取值，零额外训练成本。
//
// 单位（三支柱 · 支柱二）：`--vocabs` 与 `--export-at` 一律是**总词表大小**（256 + K），
//       与 `train --vocab` / `cut --vocab` 完全同口径。旧的"merges 数"口径已废弃，
//       表格里 K 与总词表**两列都打印**，不再需要心算。
//
// 产出：bytes/token（越高越好）随规模的曲线 + 边际收益，用于定位拐点；
//       `--export-at` 可直接吐出对应模型，省掉"看表 → 手算 256+K → 敲 cut"这一步。

use super::cut::cut_loaded;
use super::util::{
    CliResult, TempFileGuard, ensure_parent, human_bytes, k_to_vocab, looks_like_k, parse_num_list,
    temp_path, vocab_to_k,
};
use crate::BBPETokenizer;
use crate::reader::CorpusReader;
use crate::tokenizer::{BpeState, LoadOptions};

pub const DEFAULT_VOCABS: [usize; 8] = [
    8_448, 16_640, 33_024, 49_408, 65_792, 98_560, 131_328, 150_256,
];

pub fn run(args: Vec<String>) -> CliResult {
    let mut model = String::new();
    let mut held_out = String::new();
    let mut format = String::from("jsonl");
    let mut vocabs: Vec<usize> = DEFAULT_VOCABS.to_vec();
    let mut export_at: Vec<usize> = Vec::new();
    let mut export_dir: Option<String> = None;
    let mut max_bytes: u64 = 50 << 20;
    let mut allow_legacy = false;

    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let mut v = |flag: &str| it.next().ok_or(format!("{flag} 需要参数"));
        match k.as_str() {
            "--model" => model = v("--model")?,
            "--held-out" => held_out = v("--held-out")?,
            "--format" => format = v("--format")?,
            "--vocabs" => vocabs = parse_num_list(&v("--vocabs")?, "--vocabs")?,
            "--export-at" => export_at = parse_num_list(&v("--export-at")?, "--export-at")?,
            "--export-dir" => export_dir = Some(v("--export-dir")?),
            "--max-bytes" => {
                max_bytes = v("--max-bytes")?
                    .parse()
                    .map_err(|e| format!("--max-bytes: {e}"))?
            }
            "--allow-legacy-pattern" => allow_legacy = true,
            other => return Err(format!("未知参数: {other}").into()),
        }
    }
    if model.is_empty() || held_out.is_empty() {
        return Err("必须提供 --model 与 --held-out".into());
    }
    for &v in vocabs.iter().chain(export_at.iter()) {
        if v <= 256 {
            return Err(format!("总词表必须 > 256（256 基础字节 + merges 数），实际 {v}").into());
        }
        if looks_like_k(v) {
            println!(
                "⚠️  本命令用总词表口径：{v} 看起来像 merges 数 K；若你指的是 K={v}，请写 {}",
                k_to_vocab(v)
            );
        }
    }

    // 载入 K_max 模型（只需一次）
    let full = BBPETokenizer::load_with_options(
        &model,
        LoadOptions {
            allow_legacy,
            validate: true,
        },
    )?;
    let total_available = full.state.merges.len();
    println!("== bbpe sweep（单次训练，扫描全部规模） ==");
    println!(
        "  模型: {} | 可用 merges {}（K_max）⟹ 最大总词表 {}",
        model,
        total_available,
        k_to_vocab(total_available)
    );
    println!(
        "  held-out: {} [{}] | 上限 {}",
        held_out,
        format,
        human_bytes(max_bytes)
    );
    println!("  单位：『总词表』= 256 基础字节 + merges 数；K 列仅供对照");
    println!();

    // n3：临时截断模型交给 RAII 守卫 —— 原实现只在循环结束后删除，
    // 而 `st.save(...)?` / `--format` 非法等提前返回会把文件遗留在 /tmp。
    let tmp = TempFileGuard::new(temp_path("bbpe_sweep", ".json"));
    let tmp_str = tmp
        .path()
        .expect("TempFileGuard 刚创建必然有路径")
        .to_string_lossy()
        .to_string();

    // 注意方向：bytes/token **越高越好**（每个 token 压进更多字节 ⟹ 同样文本需要更少 token）。
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "总词表", "K", "bytes/tok", "汉字/tok~", "相对最小提升", "边际(每+1K)", "备注"
    );
    let mut first_bpt: Option<f64> = None;
    let mut prev_bpt: Option<f64> = None;
    let mut prev_k: Option<usize> = None;

    for &vocab in &vocabs {
        // 总词表 → merges 数；超出 K_max 自动钳制（truncate_to_k 内 min）
        let k = vocab_to_k(vocab).min(total_available);

        // 按 id 截断：new_id = 256 + i ⟹ 保留前 k 次合并（贪心前缀性质）。
        // m6：改走 `cut_loaded` —— 直接 `truncate_to_k` + `save` 会让落盘模型的
        // meta.train.actual_merges 仍是 K_max 的训练值（inspect 展示失真）。
        // cut_loaded 同时修正 actual_merges 并记录 truncated_from 出身。
        let mut st = full.state.clone();
        cut_loaded(&mut st, &tmp_str, k_to_vocab(k))?;
        let actual_vocab = st.vocab.len();
        let tok = BBPETokenizer::load(&tmp_str)?;

        // 在 held-out 上编码，统计 bytes/token
        let reader = match format.as_str() {
            "jsonl" => CorpusReader::JsonlStaticText {
                filepath: held_out.clone(),
            },
            "text" => CorpusReader::Text {
                filepath: held_out.clone(),
            },
            // n3：清理已由 tmp 守卫负责
            other => return Err(format!("未知 --format: {other}").into()),
        };
        let mut bytes: u64 = 0;
        let mut tokens: u64 = 0;
        // R4-7：held-out 读取也走严格迭代器（IO 错误不再被当 EOF，否则
        // bytes/token 会按截断语料计算，静默失真）
        for line in reader.iter_lines_checked(None)? {
            let line = line?;
            bytes += line.len() as u64;
            tokens += tok.encode_ordinary(&line).len() as u64;
            if bytes >= max_bytes {
                break;
            }
        }
        let bpt = bytes as f64 / tokens.max(1) as f64;
        if first_bpt.is_none() {
            first_bpt = Some(bpt);
        }
        let rel = (bpt - first_bpt.unwrap()) / first_bpt.unwrap() * 100.0;
        let marg = match (prev_bpt, prev_k) {
            (Some(p), Some(pk)) if k > pk => (bpt - p) / p * 100.0 / ((k - pk) as f64 / 1000.0),
            _ => f64::NAN,
        };
        let note = if k >= total_available { "= K_max" } else { "" };
        println!(
            "{:>10} {:>10} {:>10.4} {:>10.2} {:>11.2}% {:>11.3}% {:>10}",
            actual_vocab,
            k,
            bpt,
            bpt / 3.0, // 粗估：中文 1 汉字 = 3 字节
            rel,
            marg,
            note
        );
        prev_bpt = Some(bpt);
        prev_k = Some(k);
    }
    // n3：临时文件的清理交给 tmp 守卫（作用域结束时自动执行）
    println!();
    println!("  读法：bytes/token **越高越好**（每 token 覆盖更多字节 ⟹ 同文本需更少 token）。");
    println!(
        "        『边际』= 每增加 1K merges 带来的压缩率相对提升，衰减到 ~0.1%/1K 以下即进入拐点区；"
    );
    println!(
        "        此时再加 K 只是线性增加 embedding 参数（+1K × d_model × 2）而几乎不改善压缩率。"
    );

    // ---------- 闭环：直接导出选定规模（不必手算 256+K 再敲 cut）----------
    let dir = export_dir.clone().unwrap_or_else(|| {
        std::path::Path::new(&model)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| ".".to_string())
    });
    let stem = std::path::Path::new(&model)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());

    if !export_at.is_empty() {
        println!();
        println!("== 导出 ==");
        for &v in &export_at {
            let out = std::path::Path::new(&dir).join(format!("{stem}_{v}.json"));
            let out_s = out.to_string_lossy().to_string();
            ensure_parent(&out_s)?;
            let mut st = BpeState::load_with_options(
                &model,
                LoadOptions {
                    allow_legacy,
                    validate: true,
                },
            )?;
            let k = cut_loaded(&mut st, &out_s, v)?;
            println!(
                "  总词表 {}（256 + {} merges）⟹ {}",
                k_to_vocab(k),
                k,
                out_s
            );
            if k_to_vocab(k) != v {
                println!(
                    "     ⚠️  源模型 merges 不足：请求 {} 被钳制到 {}（K_max = {}）",
                    v,
                    k_to_vocab(k),
                    total_available
                );
            }
        }
        println!("  提示：也可直接 bbpe cut --model {model} --vocab <总词表> --output <路径>");
    } else {
        // 不导出也给出可直接复制的命令，杜绝单位心算
        println!();
        println!("  下一步（复制即用，单位已换算好）：");
        for &v in &vocabs {
            if v <= k_to_vocab(total_available) {
                println!(
                    "    bbpe cut --model {model} --vocab {v} --output {}/{stem}_{v}.json",
                    dir
                );
            }
        }
        let list = vocabs
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!("    或加 --export-at {list} 让本命令直接导出");
    }
    Ok(())
}

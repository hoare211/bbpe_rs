// src/cli/encode.rs
//
// 用已训练模型编码文本（`bbpe encode`）。
//
// 为什么需要它（2026-10-12 评审 C2）：训练链（train / sweep / cut / inspect /
// compare）完整覆盖了"产出模型"，但"模型训完到底编出来什么样"此前只能走 Rust
// API 或 Python 绑定 —— CLI 用户没有任何入口验证训练产物。本命令补上
// "训练产物 → 使用"的最后一环，同时是最直接的端到端回归手段（golden 覆盖的是
// merges 表，不是"该模型在此文本上的编码输出"）。
//
// 特殊 token 语义与 Rust 核心（tokenizer.rs::encode）逐字一致：
//   - 默认一个 special 都不放行（与 encode_ordinary 同语义，防注入）；
//   - `--allowed-special a,b` 白名单逐项放行；任一项为 "all" 即放行全部
//     （tiktoken 式约定，C4；与 Python 绑定 resolve_allowed_special 对齐）；
//   - `--all-special` 直接全放行。
//
// 注意（C5 观察的显式化）：encode 只接受合法 UTF-8 文本（`&str`），
// 非法字节序列必须在调用前自行处理（from_utf8_lossy 会引入 U+FFFD，改变序列）。

use super::util::CliResult;
use crate::tokenizer::{BBPETokenizer, LoadOptions};
use std::collections::HashSet;

/// token 明细的默认打印上限：ids 一行全给（回归/管道消费），
/// 逐 token 表格默认只列前 N 条，完整序列走 `--json`。
const TOKEN_PREVIEW: usize = 64;

pub fn run(args: Vec<String>) -> CliResult {
    let mut model = String::new();
    let mut text_arg: Option<String> = None;
    let mut file: Option<String> = None;
    let mut allowed_special: Vec<String> = Vec::new();
    let mut all_special = false;
    let mut json = false;
    let mut allow_legacy = false;

    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let mut v = |flag: &str| it.next().ok_or(format!("{flag} 需要参数"));
        match k.as_str() {
            "--model" => model = v("--model")?,
            "--text" => text_arg = Some(v("--text")?),
            "--file" => file = Some(v("--file")?),
            // 逗号分隔；空项忽略（与 util::parse_num_list 同口径）
            "--allowed-special" => {
                allowed_special = v("--allowed-special")?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            "--all-special" => all_special = true,
            "--json" => json = true,
            "--allow-legacy-pattern" => allow_legacy = true,
            other => return Err(format!("未知参数: {other}").into()),
        }
    }
    if model.is_empty() {
        return Err("必须提供 --model".into());
    }

    let text = match (text_arg, file) {
        (Some(_), Some(_)) => return Err("--text 与 --file 只能二选一".into()),
        (Some(t), None) => t,
        (None, Some(f)) => std::fs::read_to_string(&f)
            .map_err(|e| format!("读 --file {f:?} 失败: {e}（encode 只接受合法 UTF-8 文本）"))?,
        (None, None) => return Err("必须提供 --text 或 --file".into()),
    };

    let tok = BBPETokenizer::load_with_options(
        &model,
        LoadOptions {
            allow_legacy,
            validate: true,
        },
    )?;

    let allowed: HashSet<&str> = allowed_special.iter().map(String::as_str).collect();
    let enc = tok.encode(&text, &allowed, all_special);

    if json {
        // 机器可读输出：ids / tokens / offsets 全量（供回归与管道消费）
        let value = serde_json::json!({
            "ids": enc.ids,
            "tokens": enc.tokens,
            "offsets": enc
                .offsets
                .iter()
                .map(|(s, e)| vec![*s, *e])
                .collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string(&value)?);
        return Ok(());
    }

    println!("== bbpe encode ==");
    println!("  模型: {model}");
    println!("  文本 {} 字节 ⟹ {} tokens", text.len(), enc.ids.len());
    println!(
        "  ids: {}",
        enc.ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("  tokens:");
    for (i, (id, t)) in enc.ids.iter().zip(&enc.tokens).enumerate() {
        if i >= TOKEN_PREVIEW {
            println!(
                "    …（其余 {} 个略；完整输出用 --json）",
                enc.ids.len() - i
            );
            break;
        }
        let (s, e) = enc.offsets[i];
        println!("    [{i:>4}] id={id:<6} bytes={s}..{e} {t:?}");
    }
    Ok(())
}

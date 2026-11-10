// src/cli/decode.rs
//
// 用已训练模型把 id 序列解码回文本（`bbpe decode`）—— `bbpe encode` 的逆操作。
//
// 与 encode 合起来闭合"训练产物 → 使用"的工具链闭环（2026-10-12 评审 C2）。
// 默认只把解码文本写到 stdout（管道友好），统计信息走 stderr ⟹
// `bbpe decode --model M --ids 1,2,3 > out.txt` 得到的就是纯文本。

use super::util::CliResult;
use crate::tokenizer::{BBPETokenizer, LoadOptions};

pub fn run(args: Vec<String>) -> CliResult {
    let mut model = String::new();
    let mut ids_arg: Option<String> = None;
    let mut json = false;
    let mut allow_legacy = false;

    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let mut v = |flag: &str| it.next().ok_or(format!("{flag} 需要参数"));
        match k.as_str() {
            "--model" => model = v("--model")?,
            "--ids" => ids_arg = Some(v("--ids")?),
            "--json" => json = true,
            "--allow-legacy-pattern" => allow_legacy = true,
            other => return Err(format!("未知参数: {other}").into()),
        }
    }
    if model.is_empty() {
        return Err("必须提供 --model".into());
    }
    let ids_arg = ids_arg.ok_or("必须提供 --ids（逗号或空格分隔）")?;

    // 逗号 / 空白均可作分隔符（命令行里 `--ids "1 2 3"` 很自然）
    let mut ids = Vec::new();
    for part in ids_arg
        .split([',', ' ', '\n', '\t'])
        .filter(|s| !s.is_empty())
    {
        let id: i32 = part
            .parse()
            .map_err(|e| format!("--ids: {part:?} 不是合法整数: {e}"))?;
        ids.push(id);
    }
    if ids.is_empty() {
        return Err("--ids 解析结果为空".into());
    }

    let tok = BBPETokenizer::load_with_options(
        &model,
        LoadOptions {
            allow_legacy,
            validate: true,
        },
    )?;

    let text = tok.decode(&ids)?;

    if json {
        let value = serde_json::json!({ "text": text, "n_tokens": ids.len() });
        println!("{}", serde_json::to_string(&value)?);
    } else {
        eprintln!(
            "bbpe decode: {} ids ⟹ {} 字节（词表 {} 项）",
            ids.len(),
            text.len(),
            tok.state.vocab.len()
        );
        // R4-26：注释与行为必须一致。解码文本**原样**写到 stdout（不加引号、
        // 不做任何转义）；末尾若无换行则补一个 —— 这是为了终端/管道友好
        // （`> out.txt` 不会把 shell 提示符粘到末行），不是内容修饰。
        // 需要严格逐字节 payload 的场景请用 `--json` 或用 Python/Rust API。
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
    }
    Ok(())
}

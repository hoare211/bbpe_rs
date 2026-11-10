// src/cli/cut.rs
//
// 从 K_max 模型导出任意 K（≤ K_max）的正式模型 —— **零额外训练成本**。
//
// 原理（与 sweep 同源）：BPE 是贪心的，第 i 次合并只依赖前 i-1 次合并的结果
// ⟹ **前 k 次合并 == 用 vocab_size = 256+k 直接训练所得的合并序列**（严格成立，非近似）。
// 而本项目 `new_id = 256 + i` 把合并次序编码进了 id ⟹ 按 id 截断即得目标词表。
//
// 单位：`--vocab` 是**总词表大小**（256 基础字节 + merges 数），与 train / sweep 同口径。
// 例：想要 100,000 个 merges ⟹ `--vocab 100256`。

use super::util::{CliResult, ensure_parent, k_to_vocab, looks_like_k, vocab_to_k};
use crate::tokenizer::{BpeState, LoadOptions};
use std::error::Error;

/// 对**已加载**的 state 做截断并保存。返回实际保留的 merges 数 k
/// （源模型不足时会被钳制到 K_max）。
///
/// R4-21：打印口径修正 —— 截断产物的真实词表项数是
/// `256 + k + special_tokens.len()`（`truncate_to_k` 会把 special 重新注入），
/// 此前按 `256 + k` 打印，与 `st.vocab.len()` 自相矛盾。
pub fn cut_loaded(st: &mut BpeState, output: &str, vocab: usize) -> Result<usize, Box<dyn Error>> {
    if vocab <= 256 {
        return Err("--vocab 是总词表大小（256 基础字节 + merges 数），必须 > 256".into());
    }
    let target_k = vocab_to_k(vocab);
    let k = st.truncate_to_k(target_k);

    // 出身信息（meta）必须继承（G1 可追溯）；把 actual_merges 修正为截断后的事实，
    // 并记录"从哪个规模截断而来"（m6）。
    if let Some(meta) = st.meta.as_mut() {
        if let Some(t) = meta.train.as_mut() {
            t.actual_merges = k;
            // 只在首次截断时记录源规模：连续多次截断应保留**最初**的训练规模，
            // 而不是上一次的中间值（否则链条一长就丢失了真实的出身）。
            if t.truncated_from.is_none() {
                t.truncated_from = Some(t.vocab_size);
            }
        }
    }
    ensure_parent(output)?;
    st.save(output)?;
    Ok(k)
}

/// 从磁盘加载 → 截断 → 保存（sweep 的 `--export-at` 也走这条路径）
pub fn cut_model(
    model: &str,
    output: &str,
    vocab: usize,
    allow_legacy: bool,
) -> Result<usize, Box<dyn Error>> {
    let mut st = BpeState::load_with_options(
        model,
        LoadOptions {
            allow_legacy,
            validate: true,
        },
    )?;
    let k = cut_loaded(&mut st, output, vocab)?;
    // R4-21：真实词表项数 = 256 基础 + k merges + n_special
    // （truncate_to_k 会把 special token 重新注入 vocab，此前按 256+K 打印
    //  与 st.vocab.len() 自相矛盾）
    println!(
        "  ⟹ 导出 {} | 词表 {} 项（256 基础 + {} merges + {} special）",
        output,
        st.vocab.len(),
        k,
        st.special_tokens.len()
    );
    Ok(k)
}

pub fn run(args: Vec<String>) -> CliResult {
    let mut model = String::new();
    let mut output = String::new();
    let mut vocab: usize = 0;
    let mut allow_legacy = false;

    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let mut v = |flag: &str| it.next().ok_or(format!("{flag} 需要参数"));
        match k.as_str() {
            "--model" => model = v("--model")?,
            "--output" => output = v("--output")?,
            "--vocab" => vocab = v("--vocab")?.parse().map_err(|e| format!("--vocab: {e}"))?,
            "--allow-legacy-pattern" => allow_legacy = true,
            other => return Err(format!("未知参数: {other}").into()),
        }
    }
    if model.is_empty() || output.is_empty() || vocab == 0 {
        return Err("必须提供 --model / --output / --vocab".into());
    }
    if looks_like_k(vocab) {
        println!(
            "⚠️  --vocab 用总词表口径：{} 看起来像 merges 数 K；若你指的是 K={}，请写 {}",
            vocab,
            vocab,
            k_to_vocab(vocab)
        );
    }

    let mut st = BpeState::load_with_options(
        &model,
        LoadOptions {
            allow_legacy,
            validate: true,
        },
    )?;
    let src_merges = st.merges.len();
    println!("== bbpe cut（按合并次序截断） ==");
    println!(
        "  源模型: {}（{} 项词表 / {} merges）",
        model,
        st.vocab.len(),
        src_merges
    );
    println!(
        "  目标总词表 {}（= 256 基础 + {} merges）",
        vocab,
        vocab_to_k(vocab)
    );

    let k = cut_loaded(&mut st, &output, vocab)?;
    println!(
        "  产出: {} | 词表 {} 项 | merges {}",
        output,
        st.vocab.len(),
        st.merges.len()
    );
    if k < vocab_to_k(vocab) {
        println!(
            "  ⚠️  源模型 merges 不足，已钳制到 {}（目标 {}）",
            k,
            vocab_to_k(vocab)
        );
    }
    println!();
    println!(
        "  等价性：前 k 次合并 == 总词表 256+k 直接训练的结果（BPE 贪心前缀性质，严格成立）。"
    );
    Ok(())
}

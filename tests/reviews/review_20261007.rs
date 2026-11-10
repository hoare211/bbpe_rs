//! 2026-10-07 评审整改的回归测试（不依赖 pyo3 的部分）。
//!
//! 覆盖本轮新发现并已修复的问题：
//! - A1：`BBPETrainer::train` 非幂等（静默产出违反 V3 的损坏模型）
//! - A2：`fancy_regex` 运行期错误被静默吞掉（整段文本无声消失）
//! - A4：`format` 字段此前无法反映文件真实值（inspect 硬编码常量）
//! - A7：`assign_shard` 低位取模导致分片负载不均
//! - M2 遗留：`end_reason = "no_more_pairs"` 此前无任何断言
//!
//! 以及此前评审指出、此前零覆盖的区域：
//! - `reader.rs` 的 `JsonlStaticConversations` / `JsonlDynamic`（含 m5 保序修复）
//! - `pre_tokenizer` 的 `max_chunk_bytes != 512` / `== 0` 分支（m4 修复无守护）
//! - 输入侧计数的守恒律
//!
//! A3（`encode_parallel` 漏改 m7）位于 pyo3 绑定层，其单元测试在
//! `src/bindings/python.rs` 的 `#[cfg(test)]` 内，Python 侧另有端到端断言。

use bbpe_rs::pre_tokenizer::{DEFAULT_PATTERN, PreTokenizer};
use bbpe_rs::reader::CorpusStats;
use bbpe_rs::tokenizer::{BpeState, LoadOptions};
use bbpe_rs::{BBPETrainer, CorpusReader, TrainerConfig};
use std::io::Write;
use std::sync::Arc;

// ============================================================
// 脚手架
// ============================================================

/// 进程隔离的临时目录（每个测试一个，不依赖 CWD —— m15 的范式）。
struct TmpDir(std::path::PathBuf);

impl TmpDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "bbpe_rv7_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
    fn text_corpus(&self, name: &str, lines: &[&str]) -> String {
        self.write_lines(name, lines.iter().copied())
    }
    /// 同上，但接受拥有所有权的 `String`（便于用 `format!` 动态生成语料）
    fn text_corpus_owned(&self, name: &str, lines: &[String]) -> String {
        self.write_lines(name, lines.iter().map(|s| s.as_str()))
    }
    fn write_lines<'a, I: Iterator<Item = &'a str>>(&self, name: &str, lines: I) -> String {
        let p = self.0.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
        p.to_string_lossy().to_string()
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// 语料 A：`the quick brown fox ...`
fn corpus_a() -> Vec<&'static str> {
    vec!["the quick brown fox jumps over the lazy dog"; 300]
}

/// 语料 B：与 A **完全不同**的高频词 —— 这是 A1 缺陷的触发条件（二次训练
/// 若不报错，残留 id 与新 new_id 撞车，V3 立刻失败）。
fn corpus_b() -> Vec<&'static str> {
    vec!["hello world foo bar baz qux quux corge grault garply waldo"; 300]
}

/// 语料 C：4000 行互异短词 —— 合并空间充足，足以把 vocab_size 撑到目标。
///
/// 实测依据：`300x fox`（去重后仅 9 个不同词）无论 vocab 设多大都会在
/// 32 次合并后 `no_more_pairs`；必须用互异词才能触发 `vocab_size_reached`。
fn corpus_rich() -> Vec<String> {
    (0..4000).map(|i| format!("w{i}q{i}e{i}r{i}")).collect()
}

fn merges_of(st: &BpeState) -> Vec<((i32, i32), i32)> {
    let mut v: Vec<_> = st.merges.iter().map(|(k, &n)| (*k, n)).collect();
    v.sort_unstable();
    v
}

// ============================================================
// A1：train 非幂等
// ============================================================

#[test]
fn a1_second_train_errors_not_corrupt_model() {
    let d = TmpDir::new("a1");
    let a = d.text_corpus("a.txt", &corpus_a());
    let b = d.text_corpus("b.txt", &corpus_b());

    let mut t = BBPETrainer::new(TrainerConfig::new(400), None).unwrap();
    t.train(&CorpusReader::Text { filepath: a }).unwrap();
    assert!(t.is_trained());
    let run1 = merges_of(&t.state);
    assert!(!run1.is_empty());
    t.state.validate().expect("第一次训练的产物必须自洽");

    // 二次训练（换语料）必须**显式失败**
    let err = t
        .train(&CorpusReader::Text { filepath: b })
        .expect_err("A1：二次训练必须报 AlreadyTrained，而不是静默产出违反 V3 的模型");
    let msg = err.to_string();
    assert!(
        msg.contains("reset"),
        "报错必须指明处置方式（reset），实际：{msg}"
    );
    assert!(
        msg.contains(&run1.len().to_string()),
        "报错应带上残留 merges 数，实际：{msg}"
    );

    // 关键：失败后 state 原样保留（不被半途写坏），仍自洽
    assert_eq!(merges_of(&t.state), run1, "失败的训练不得改动已有 state");
    t.state.validate().expect("失败的训练不得污染 state");
}

#[test]
fn a1_retrain_after_reset_matches_fresh_instance() {
    let d = TmpDir::new("a1reset");
    let a = d.text_corpus("a.txt", &corpus_a());
    let b = d.text_corpus("b.txt", &corpus_b());

    // 路线 1：train → reset → train
    let mut t = BBPETrainer::new(TrainerConfig::new(400), None).unwrap();
    t.train(&CorpusReader::Text { filepath: a }).unwrap();
    t.reset();
    assert!(!t.is_trained(), "reset 后 is_trained 应为 false");
    t.train(&CorpusReader::Text {
        filepath: b.clone(),
    })
    .unwrap();

    // 路线 2：全新实例只训 B
    let mut fresh = BBPETrainer::new(TrainerConfig::new(400), None).unwrap();
    fresh.train(&CorpusReader::Text { filepath: b }).unwrap();

    // 两者必须逐条相同 —— 这正是 A1 修复前被破坏的不变量
    assert_eq!(
        merges_of(&t.state),
        merges_of(&fresh.state),
        "reset() 后的训练结果必须与全新 trainer 等价（A1 修复的核心不变量）"
    );
    t.state.validate().expect("reset 后重训的产物必须自洽");
}

#[test]
fn a1_reset_preserves_special_tokens_layout() {
    let d = TmpDir::new("a1sp");
    let a = d.text_corpus("a.txt", &corpus_a());
    let mut cfg = TrainerConfig::new(400);
    cfg.special_tokens = vec!["<|a|>".to_string(), "<|b|>".to_string()];
    let mut t = BBPETrainer::new(cfg, None).unwrap();
    t.train(&CorpusReader::Text { filepath: a }).unwrap();
    let before: Vec<_> = {
        let mut v: Vec<_> = t
            .state
            .special_tokens
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        v.sort_unstable();
        v
    };
    t.reset();
    let after: Vec<_> = {
        let mut v: Vec<_> = t
            .state
            .special_tokens
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(
        before, after,
        "reset 必须按 new() 的同一规则重建 special token 尾部布局"
    );
    // 基础字节必须完好
    for i in 0..256 {
        assert_eq!(
            t.state.vocab.get(&i),
            Some(&vec![i as u8]),
            "reset 后基础字节 {i} 缺失"
        );
    }
    t.state.validate().expect("reset 后的 state 必须自洽");
}

#[test]
fn a1_new_trainer_is_untrained() {
    let t = BBPETrainer::new(TrainerConfig::new(400), None).unwrap();
    assert!(!t.is_trained());
    assert!(t.state.merges.is_empty());
}

// ============================================================
// A2：正则运行期错误不再静默吞掉
// ============================================================

#[test]
fn a2_uncovered_gap_errors_not_silently_dropped() {
    // A2 的**真正机制**（初版评审归因错误，已实证纠正）：
    // `find_iter` 只产出匹配，匹配之间的"间隙"永不进入回调。只要 pattern 不是
    // 全函数，未匹配部分就被静默丢弃 —— 且**不是**因为 Err 被吞。
    //
    // 实证：fancy_regex 0.18 对灾难性回溯返回 `Ok`（零匹配）而非 `Err`
    // （`(a+)+$` 配 5000 个 a + b 仍返回 Ok，chunks=0），所以 clippy 那条
    // `unnecessary if let` 指向的并不是本问题的主因。
    let pt = PreTokenizer::new(r"\s+").unwrap();

    let mut out = Vec::new();
    let err = pt
        .tokenize_with_checked("你好世界", |c, off| out.push((c.len(), off)))
        .expect_err("覆盖不全的 pattern 必须报错，而不是静默把整行丢掉");
    let msg = err.to_string();
    assert!(
        msg.contains("未覆盖"),
        "报错须指明是未覆盖间隙，实际：{msg}"
    );
    assert!(
        msg.contains("必须覆盖输入的**每一个字符**"),
        "报错须给出可操作的方向，实际：{msg}"
    );
    assert!(
        out.is_empty(),
        "「你好世界」对空白 pattern 无任何匹配，不应产出 chunk"
    );
}

#[test]
fn a2_trailing_gap_and_no_match_detected() {
    // 中间有匹配、尾部还有未覆盖内容（最容易被忽略的一类）
    let pt = PreTokenizer::new(r"[0-9]+").unwrap();
    let err = pt
        .tokenize_with_checked("123abc", |_, _| {})
        .expect_err("尾部间隙必须被检出");
    assert!(err.to_string().contains("未覆盖"), "实际：{err}");

    // 无任何匹配 ⟹ 整个输入都是间隙
    let err2 = pt
        .tokenize_with_checked("abc", |_, _| {})
        .expect_err("全无匹配也必须报错");
    assert!(err2.to_string().contains("未覆盖"), "实际：{err2}");
}

#[test]
fn a2_runtime_error_surfaces_and_is_deterministic() {
    // 次要但仍需守护：正则的运行期 Err（回溯超限等）此前被 `if let Ok(..)` 丢弃。
    // 验证：要么成功、要么报可诊断的错，绝不 panic，也绝不静默给出错误结果。
    let pt = PreTokenizer::new(r"(a+)+$").unwrap();
    let mut line = "a".repeat(5000);
    line.push('b');
    let mut chunks = 0usize;
    match pt.tokenize_with_checked(&line, |_, _| chunks += 1) {
        Ok(()) => {
            // fancy_regex 0.18 实测返回 Ok（零匹配）—— 这是"正则确实不匹配"，
            // 与"错误被吞"是两回事，故此处不断言错误，只要求语义稳定。
            assert_eq!(chunks, 0, "连续 a 加 b 对该 pattern 确实无匹配");
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("预分词"), "若报错则须可诊断，实际：{msg}");
        }
    }
}

#[test]
fn a2_overlapping_or_empty_match_rejected() {
    // 可匹配空串或自重叠的 pattern 会让 find_iter 产出重叠匹配 ⟹ 偏移错乱
    let pt = PreTokenizer::new(r"a*").unwrap();
    let err = pt
        .tokenize_with_checked("aab", |_, _| {})
        .expect_err("重叠/空匹配必须被检出");
    let msg = err.to_string();
    assert!(
        msg.contains("重叠") || msg.contains("未覆盖"),
        "报错须指明重叠或未覆盖，实际：{msg}"
    );
}

#[test]
fn a2_default_pattern_is_total() {
    // 把各类"非字母数字"边界字符都试一遍，每个 chunk 拼起来必须**逐字节等于**输入。
    let pt = PreTokenizer::new(DEFAULT_PATTERN).unwrap();
    let cases: Vec<String> = vec![
        "你好世界".into(),
        "a\u{1F600}b".into(), // emoji（4 字节，非 BMP）
        "a\u{7}b".into(),     // 控制字符
        "a\u{200B}b".into(),  // 零宽空格
        "\u{FEFF}abc".into(), // BOM
        "e\u{301}".into(),    // 组合记号
        "a\u{3000}b".into(),  // 全角空格
        "\u{20000}".into(),   // CJK 扩展 B（4 字节）
        "\u{80}".into(),      // 孤立 UTF-8 延续字节
        "a\u{200F}b".into(),  // RTL 标记
        "a\u{0}b".into(),     // NUL
        "Mixed中英123数字!标点?。".into(),
    ];
    for s in cases {
        let mut joined = Vec::new();
        pt.tokenize_with_checked(&s, |c, _| joined.extend_from_slice(c))
            .unwrap_or_else(|e| panic!("默认 pattern 不应失败，输入 {s:?}: {e}"));
        assert_eq!(
            joined,
            s.as_bytes(),
            "默认 pattern 漏掉了字节：输入 {:?}，拼回 {:?}",
            s,
            String::from_utf8_lossy(&joined)
        );
    }
}

#[test]
fn a2_training_hard_fails_on_partial_pattern() {
    let d = TmpDir::new("a2train");
    let corpus = d.text_corpus("c.txt", &["the quick brown fox", "hello world"]);
    // `[0-9]+` 只匹配数字 ⟹ 整行文本全部落进"未覆盖间隙"
    let mut cfg = TrainerConfig::new(300);
    cfg.pre_tokenizer_pattern = r"[0-9]+".to_string();
    let mut t = BBPETrainer::new(cfg, None).unwrap();

    let err = t
        .train(&CorpusReader::Text { filepath: corpus })
        .expect_err("A2：预分词失败必须让训练终止，不能静默丢语料后继续训练");
    assert!(
        err.to_string().contains("预分词"),
        "报错应指明是预分词阶段，实际：{err}"
    );
    assert!(
        t.state.merges.is_empty(),
        "训练失败后不得留下半成品 merges（实际 {} 条）",
        t.state.merges.len()
    );
}

// ============================================================
// pre_tokenizer：max_chunk_bytes 的三个分支（m4 修复此前无任何守护）
// ============================================================

#[test]
fn m4_max_chunk_bytes_three_branches() {
    let text = "中文字".repeat(20); // 60 个汉字 = 180 字节

    // 分支 1：默认 512 —— 180 < 512，不切分
    let pt = PreTokenizer::new(DEFAULT_PATTERN).unwrap();
    assert_eq!(pt.max_chunk_bytes(), 512);
    let mut n = 0;
    pt.tokenize_with(&text, |c, _| {
        n += 1;
        assert!(c.len() <= 512);
    });
    assert_eq!(n, 1, "180 字节不应被 512 上限切开");

    // 分支 2：max_chunk_bytes = 3（恰好一个汉字）—— 按 char boundary 切
    let pt3 = PreTokenizer::with_max_chunk_bytes(DEFAULT_PATTERN, 3).unwrap();
    assert_eq!(pt3.max_chunk_bytes(), 3);
    let mut got = Vec::new();
    pt3.tokenize_with(&text, |c, off| {
        got.push((String::from_utf8_lossy(c).into_owned(), off))
    });
    assert_eq!(got.len(), 60, "60 个汉字应切成 60 块，实际 {}", got.len());
    // 每个 chunk 恰好 1 个汉字 = 3 字节 ⟹ 第 i 块应是原文的第 i 个汉字
    for (i, (chunk, off)) in got.iter().enumerate() {
        let expect = &text[i * 3..i * 3 + 3];
        assert_eq!(chunk.as_str(), expect, "第 {i} 块内容错：{chunk:?}");
        assert_eq!(*off, i * 3, "第 {i} 块偏移错：{off}");
    }

    // 分支 3：max_chunk_bytes = 0 —— 语义是"不切分"
    let pt0 = PreTokenizer::with_max_chunk_bytes(DEFAULT_PATTERN, 0).unwrap();
    let mut one = Vec::new();
    pt0.tokenize_with(&text, |c, _| one.push(c.len()));
    assert_eq!(one, vec![180], "max_chunk_bytes=0 应不切分");
}

#[test]
fn m4_offsets_align_with_original() {
    let text = "中文字abc中文字";
    let pt = PreTokenizer::with_max_chunk_bytes(DEFAULT_PATTERN, 4).unwrap();
    let mut rebuilt = Vec::new();
    pt.tokenize_with(text, |c, off| {
        // 断言：该 chunk 确实是原文 [off, off+len) 的切片
        assert_eq!(
            &text.as_bytes()[off..off + c.len()],
            c,
            "偏移 {off} 与内容不符"
        );
        rebuilt.extend_from_slice(c);
    });
    assert_eq!(rebuilt, text.as_bytes(), "偏移切分后拼回应等于原文");
}

// ============================================================
// reader：JsonlStaticConversations（此前 Rust 侧零覆盖）
// ============================================================

#[test]
fn reader_static_conversations_join_and_skip() {
    let d = TmpDir::new("rd_conv");
    let p = d.path().join("conv.jsonl");
    let mut f = std::fs::File::create(&p).unwrap();
    writeln!(
        f,
        r#"{{"conversations":[{{"content":"你好"}},{{"content":"世界"}}]}}"#
    )
    .unwrap();
    writeln!(
        f,
        r#"{{"conversations":[{{"role":"u"}},{{"content":"缺role也行"}},{{"content":"尾"}}]}}"#
    )
    .unwrap();
    writeln!(f, r#"{{"conversations":[]}}"#).unwrap(); // 拼接为空 ⟹ skipped
    writeln!(f, r#"{{"no_conversations":1}}"#).unwrap(); // 缺字段 ⟹ skipped
    writeln!(f, r#"这不是 JSON"#).unwrap(); // 解析失败 ⟹ skipped
    writeln!(f).unwrap(); // 空行 ⟹ 不计 skipped
    writeln!(f, r#"{{"conversations":[{{"content":"尾行"}}]}}"#).unwrap();
    drop(f);

    let stats = Arc::new(CorpusStats::default());
    let r = CorpusReader::JsonlStaticConversations {
        filepath: p.to_string_lossy().to_string(),
        sep: "\n".to_string(),
    };
    let lines: Vec<String> = r.iter_lines_with_stats(stats.clone()).unwrap().collect();

    assert_eq!(lines[0], "你好\n世界", "conversations 应按 sep 拼接");
    assert_eq!(lines[1], "缺role也行\n尾", "缺 role 的消息也应取 content");
    assert_eq!(lines[2], "尾行");
    assert_eq!(lines.len(), 3, "只应产出 3 行");

    let (total_lines, _b, skipped, _r) = stats.snapshot();
    assert_eq!(total_lines, 7, "物理行数应为 7（空行计入）");
    assert_eq!(
        skipped, 3,
        "空 conversations / 缺字段 / 坏 JSON 各计 1 次 skipped"
    );
}

// ============================================================
// reader：JsonlDynamic（此前 Rust 侧零覆盖；m5 保序修复无守护）
// ============================================================

#[test]
fn reader_dynamic_multi_field_declaration_order() {
    let d = TmpDir::new("rd_dyn");
    let p = d.path().join("dyn.jsonl");
    let mut f = std::fs::File::create(&p).unwrap();
    writeln!(f, r#"{{"a":"AAA","b":"BBB","c":"CCC"}}"#).unwrap();
    writeln!(f, r#"{{"a":123,"b":"BB2","c":"CC2"}}"#).unwrap(); // a 非字符串 ⟹ a 分支不产出
    writeln!(f, r#"{{"z":"ZZZ"}}"#).unwrap(); // 全缺 ⟹ skipped
    writeln!(f, r#"彻底不是 JSON"#).unwrap(); // 坏行 ⟹ skipped
    drop(f);

    let stats = Arc::new(CorpusStats::default());
    let r = CorpusReader::JsonlDynamic {
        filepath: p.to_string_lossy().to_string(),
        schema_rules: vec![
            (vec!["a".to_string()], "str".to_string()),
            (vec!["b".to_string()], "str".to_string()),
            (vec!["c".to_string()], "str".to_string()),
        ],
    };
    let lines: Vec<String> = r.iter_lines_with_stats(stats.clone()).unwrap().collect();

    // m5 守护点：同一行内多个字段的产出必须**按 schema 声明顺序**
    assert_eq!(
        &lines[0..3],
        ["AAA", "BBB", "CCC"],
        "多字段产出顺序必须 == schema 声明顺序"
    );
    assert_eq!(
        &lines[3..5],
        ["BB2", "CC2"],
        "a 为数字时 a 分支不产出，b/c 照常"
    );
    assert_eq!(lines.len(), 5);

    let (total_lines, _b, skipped, _r) = stats.snapshot();
    assert_eq!(total_lines, 4);
    assert_eq!(skipped, 2, "全缺字段行 + 坏 JSON 各计 1 次 skipped");
}

#[test]
fn reader_dynamic_type_branches() {
    let d = TmpDir::new("rd_types");
    let p = d.path().join("t.jsonl");
    std::fs::write(
        &p,
        "{\"i\":42,\"f\":1.5,\"b\":true,\"li\":[7,8],\"ls\":[\"p\",\"q\"],\"s\":\"str\"}\n",
    )
    .unwrap();

    // (字段名, 该字段使用的类型名, 期望产出)
    let cases: Vec<(&str, &str, Vec<&str>)> = vec![
        ("s", "str", vec!["str"]),
        ("i", "int", vec!["42"]),
        ("f", "float", vec!["1.5"]),
        ("b", "bool", vec!["true"]),
        ("li", "list_int", vec!["7", "8"]),
        ("ls", "list_str", vec!["p", "q"]),
    ];
    for (field, ty, expect) in cases {
        let r = CorpusReader::JsonlDynamic {
            filepath: p.to_string_lossy().to_string(),
            schema_rules: vec![(vec![field.to_string()], ty.to_string())],
        };
        let lines: Vec<String> = r.iter_lines().unwrap().collect();
        assert_eq!(lines, expect, "字段 {field} 的取值不对");
    }

    // 未知类型名必须被忽略（不产出、也不误计 skipped）
    let stats = Arc::new(CorpusStats::default());
    let r = CorpusReader::JsonlDynamic {
        filepath: p.to_string_lossy().to_string(),
        schema_rules: vec![(vec!["s".to_string()], "no_such_type".to_string())],
    };
    assert!(
        r.iter_lines_with_stats(stats.clone())
            .unwrap()
            .next()
            .is_none()
    );
    let (_, _, skipped, _) = stats.snapshot();
    assert_eq!(skipped, 1, "未知类型 ⟹ 该行无产出 ⟹ 计 1 次 skipped");
}

#[test]
fn reader_nested_path_extraction() {
    let d = TmpDir::new("rd_nest");
    let p = d.path().join("n.jsonl");
    std::fs::write(&p, "{\"data\":{\"text\":\"深层文本\"}}\n").unwrap();
    let r = CorpusReader::JsonlDynamic {
        filepath: p.to_string_lossy().to_string(),
        schema_rules: vec![(
            vec!["data".to_string(), "text".to_string()],
            "str".to_string(),
        )],
    };
    let lines: Vec<String> = r.iter_lines().unwrap().collect();
    assert_eq!(lines, vec!["深层文本"], "多级路径应能定位到字段");
}

// ============================================================
// reader：计数守恒（m14 的加强版）
// ============================================================

#[test]
fn reader_line_count_invariant() {
    let d = TmpDir::new("rd_cons");
    let p = d.path().join("c.jsonl");
    let mut f = std::fs::File::create(&p).unwrap();
    for w in ["alpha", "beta", "gamma", "delta", "epsilon"] {
        writeln!(f, "{{\"text\":\"{w}\"}}").unwrap();
    }
    writeln!(f).unwrap(); // 空行（不产出、不 skipped）
    writeln!(f).unwrap(); // 空行
    writeln!(f, "{{\"text\":123}}").unwrap(); // text 类型错 ⟹ skipped
    writeln!(f, "坏行").unwrap(); // 解析失败 ⟹ skipped
    drop(f);

    let readers = vec![
        (
            "JsonlStaticText",
            CorpusReader::JsonlStaticText {
                filepath: p.to_string_lossy().to_string(),
            },
        ),
        (
            "JsonlDynamic",
            CorpusReader::JsonlDynamic {
                filepath: p.to_string_lossy().to_string(),
                schema_rules: vec![(vec!["text".to_string()], "str".to_string())],
            },
        ),
    ];
    for (name, r) in readers {
        let stats = Arc::new(CorpusStats::default());
        let n = r.iter_lines_with_stats(stats.clone()).unwrap().count() as u64;
        let (total, _b, skipped, _rp) = stats.snapshot();
        assert_eq!(total, 9, "{name}：5 有效 + 2 空行 + 2 坏行 = 9 物理行");
        assert_eq!(n, 5, "{name}：应产出 5 行");
        assert_eq!(skipped, 2, "{name}：类型错 + 解析失败各计 1");
        // 不变量的准确形式：**空行既不产出也不计 skipped，但计入 total_lines**。
        //   total = 产出 + skipped + 空行
        // 这一点值得固化：`meta.corpus.total_lines` 若被误读成"有效行数"，
        // 就会把"空行占比"错误地算进"坏行率"。
        assert_eq!(
            total,
            n + skipped + 2,
            "{name}：计数不守恒 total={total} n={n} skipped={skipped}"
        );
    }
}

// ============================================================
// M2 遗留：end_reason 的两条分支（此前无 Rust 断言）
// ============================================================

/// M2 新增的 `no_more_pairs`：候选堆与 global_counts 双双耗尽。
///
/// 触发条件（实测）：min_frequency 必须 **= 1**，否则会先被
/// `min_frequency_terminated` 拦下 —— 最高频 pair 只有 1 次，达不到 m=2。
#[test]
fn m2_end_reason_no_more_pairs() {
    let d = TmpDir::new("m2");
    let c = d.text_corpus("c.txt", &["ab"]);
    let mut cfg = TrainerConfig::new(260);
    cfg.min_frequency = 1; // 否则 (a,b) 频次 1 < 2，会走 min_frequency_terminated
    let mut t = BBPETrainer::new(cfg, None).unwrap();
    t.train(&CorpusReader::Text { filepath: c }).unwrap();

    let tr = t.state.meta.as_ref().unwrap().train.as_ref().unwrap();
    assert_eq!(
        tr.end_reason, "no_more_pairs",
        "语料只有 \"ab\" 一行、可合并 pair 耗尽 ⟹ 必须是 no_more_pairs（M2 新增取值）"
    );
    assert_eq!(tr.actual_merges, 1, "只应完成 1 次合并（a+b）");
    t.state.validate().expect("产物必须自洽");
}

/// 目标词表被撑满时 `vocab_size_reached`（即"推论 7"成立的前提）。
#[test]
fn m2_end_reason_vocab_size_reached() {
    let d = TmpDir::new("m2b");
    let c = d.text_corpus_owned("c.txt", &corpus_rich());
    let mut t = BBPETrainer::new(TrainerConfig::new(300), None).unwrap();
    t.train(&CorpusReader::Text { filepath: c }).unwrap();
    let tr = t.state.meta.as_ref().unwrap().train.as_ref().unwrap();
    assert_eq!(tr.end_reason, "vocab_size_reached");
    assert_eq!(tr.actual_merges, 44, "300 - 256 = 44 次合并");
    t.state.validate().expect("产物必须自洽");
}

/// 达不到 min_frequency 阈值时早停。
#[test]
fn m2_end_reason_min_frequency_terminated() {
    let d = TmpDir::new("m2c");
    // 语料很小 + min_frequency 很高 ⟹ 最高频 pair 也达不到阈值
    let c = d.text_corpus("c.txt", &["ab ac ad"]);
    let mut cfg = TrainerConfig::new(300);
    cfg.min_frequency = 1000;
    let mut t = BBPETrainer::new(cfg, None).unwrap();
    t.train(&CorpusReader::Text { filepath: c }).unwrap();
    let tr = t.state.meta.as_ref().unwrap().train.as_ref().unwrap();
    assert_eq!(tr.end_reason, "min_frequency_terminated");
    assert!(tr.actual_merges < 44);
}

// ============================================================
// A4：format 字段必须反映文件真实值
// ============================================================

#[test]
fn a4_load_records_real_format() {
    let d = TmpDir::new("a4");
    let c = d.text_corpus("c.txt", &corpus_a());
    let mut cfg = TrainerConfig::new(300);
    cfg.special_tokens = vec!["<|t0|>".to_string()];
    let mut t = BBPETrainer::new(cfg, None).unwrap();
    t.train(&CorpusReader::Text { filepath: c }).unwrap();
    let m = d.path().join("m.json");
    t.save(m.to_str().unwrap()).unwrap();

    // (1) v1 新模型：两个字段都在
    let st = BpeState::load(m.to_str().unwrap()).unwrap();
    assert_eq!(st.format.as_deref(), Some("bbpe-rs-model"));
    assert_eq!(st.format_version, Some(1));

    // (2) v0 旧模型：两个字段都缺 ⟹ 都必须是 None
    //     （修复前 BpeState 根本不记 format，inspect 只能打常量，于是把
    //      v0 旧模型显示成 `format: bbpe-rs-model` 并退出码 0）
    let v0 = d.path().join("v0.json");
    let mut vocab = std::collections::BTreeMap::new();
    for i in 0..256usize {
        vocab.insert(i.to_string(), vec![i as u8]);
    }
    vocab.insert("256".to_string(), b"<|t0|>".to_vec());
    std::fs::write(
        &v0,
        serde_json::json!({
            "vocab": vocab,
            "merges": {},
            "added_tokens": {"<|t0|>": 256},
            "pre_tokenizer_pattern": DEFAULT_PATTERN,
        })
        .to_string(),
    )
    .unwrap();
    let st0 = BpeState::load_with_options(
        v0.to_str().unwrap(),
        LoadOptions {
            allow_legacy: false,
            validate: true,
        },
    )
    .unwrap();
    assert_eq!(st0.format, None, "v0 模型不应凭空声称自己是 bbpe-rs-model");
    assert_eq!(st0.format_version, None);
}

#[test]
fn a4_format_without_version_errors() {
    let d = TmpDir::new("a4h");
    let mut vocab = std::collections::BTreeMap::new();
    for i in 0..256usize {
        vocab.insert(i.to_string(), vec![i as u8]);
    }
    let p = d.path().join("half.json");
    std::fs::write(
        &p,
        serde_json::json!({
            "format": "bbpe-rs-model",   // 有 format
            "vocab": vocab,                // 缺 format_version ⟹ 契约违规
            "merges": {},
            "pre_tokenizer_pattern": DEFAULT_PATTERN,
        })
        .to_string(),
    )
    .unwrap();
    let err = match BpeState::load(p.to_str().unwrap()) {
        Ok(_) => panic!("半新半旧（有 format 缺 format_version）必须报错"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("format_version"),
        "报错应指明缺 format_version，实际：{err}"
    );
}

// ============================================================
// A5：meta 字段往返（profile 等）
// ============================================================

#[test]
fn a5_profile_round_trips() {
    use bbpe_rs::tokenizer::ModelMeta;
    let d = TmpDir::new("a5");
    let c = d.text_corpus("c.txt", &corpus_a());
    let mut t = BBPETrainer::new(TrainerConfig::new(300), None).unwrap();
    t.train(&CorpusReader::Text { filepath: c }).unwrap();
    // 模拟 CLI / Python 绑定写入 profile
    t.state.meta_mut().profile = Some("v2".to_string());

    let p = d.path().join("m.json");
    t.save(p.to_str().unwrap()).unwrap();
    let st = BpeState::load(p.to_str().unwrap()).unwrap();
    assert_eq!(
        st.meta.as_ref().unwrap().profile.as_deref(),
        Some("v2"),
        "profile 必须能落盘并读回（此前 Python 写、CLI 不写、inspect 不展示）"
    );

    // 未知 meta 字段必须被忽略（ADR-4：加载器 MUST 忽略未知字段）
    let raw = std::fs::read_to_string(&p).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    v["meta"]["totally_unknown_field"] = serde_json::json!("should be ignored");
    let p2 = d.path().join("m2.json");
    std::fs::write(&p2, v.to_string()).unwrap();
    let st2 = BpeState::load(p2.to_str().unwrap()).expect("未知 meta 字段必须被忽略而非报错");
    assert_eq!(st2.meta.as_ref().unwrap().profile.as_deref(), Some("v2"));

    // ModelMeta::default() 不应凭空造出 profile
    assert_eq!(ModelMeta::default().profile, None);
}

// ============================================================
// A7：分片散列均匀性
// ============================================================

fn fnv(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// 复刻 trainer 修复后的 assign_shard：完整 64 位哈希直接取模。
fn assign_shard_fixed(bytes: &[u8], n: usize) -> usize {
    (fnv(bytes) % n as u64) as usize
}

/// 曾经的"错误修法"：乘法散列取哈希**高位**。FNV-1A 的高位雪崩很差，
/// 顺序输入下 n=64 偏斜高达 +79%，**比修复前更糟**。保留作反例。
fn assign_shard_mulshift(bytes: &[u8], n: usize) -> usize {
    (((fnv(bytes) as u128) * (n as u128)) >> 64) as usize
}

/// 修复前的实现（在已截断到低 8 位的值上取模），用于对照。
fn assign_shard_old(bytes: &[u8], n: usize) -> usize {
    ((fnv(bytes) as usize) & 255) % n
}

#[test]
fn a7_shards_near_uniform_for_any_num_shards() {
    const N: usize = 60_000;
    let words: Vec<String> = (0..N).map(|i| format!("word{i}")).collect();

    for n in [1usize, 2, 6, 8, 10, 12, 16, 24, 32, 64] {
        let mut cnt = vec![0usize; n];
        for w in &words {
            cnt[assign_shard_fixed(w.as_bytes(), n)] += 1;
        }
        let avg = N as f64 / n as f64;
        let max = *cnt.iter().max().unwrap() as f64;
        assert!(
            max / avg < 1.08,
            "n={n} 偏斜过大：max={max} avg={avg:.1} 偏斜 {:+.1}%",
            (max / avg - 1.0) * 100.0
        );
    }
}

/// 一组"顺序化"的 key —— 对哈希的高位与低位都是最难的情况。
fn skewed_keys() -> Vec<String> {
    (0..60_000).map(|i| format!("word{i}")).collect()
}

fn max_skew(keys: &[String], n: usize, f: fn(&[u8], usize) -> usize) -> f64 {
    let mut cnt = vec![0usize; n];
    for w in keys {
        cnt[f(w.as_bytes(), n)] += 1;
    }
    let avg = keys.len() as f64 / n as f64;
    (*cnt.iter().max().unwrap() as f64 / avg - 1.0) * 100.0
}

#[test]
fn a7_fix_reduces_skew_for_non_power_of_two() {
    // 实测基线（6 万个 word{i}）：修复前 n=12 偏斜 +4.1%、n=24 偏斜 +4.6%
    let keys = skewed_keys();
    for n in [6usize, 10, 12, 24] {
        let old = max_skew(&keys, n, assign_shard_old);
        let new = max_skew(&keys, n, assign_shard_fixed);
        assert!(
            new <= old + 0.5,
            "n={n}：修复后偏斜 {new:+.2}% 不应明显劣于修复前 {old:+.2}%"
        );
    }
}

#[test]
fn a7_multiplicative_hash_worse_on_sequential() {
    // 固化这个教训：乘法散列取 FNV-1A 的**高位**，而 FNV-1A 高位雪崩很差，
    // 顺序输入下偏斜远超修复前 ⟹ 当年的"优化"会把负载均衡做得更糟。
    let keys = skewed_keys();
    for n in [2usize, 12, 24, 64] {
        let old = max_skew(&keys, n, assign_shard_old);
        let mul = max_skew(&keys, n, assign_shard_mulshift);
        let fixed = max_skew(&keys, n, assign_shard_fixed);
        assert!(
            mul > old,
            "n={n}：乘法散列偏斜 {mul:+.1}% 本就应劣于修复前 {old:+.1}%（反例前提）"
        );
        assert!(
            fixed < mul,
            "n={n}：修复后 {fixed:+.1}% 应显著优于乘法散列 {mul:+.1}%"
        );
    }
}

#[test]
fn a7_shard_assignment_is_deterministic() {
    // 同样的输入必须永远落到同一个分片（否则破坏可复现性）
    for n in [8usize, 12, 24] {
        for w in ["hello", "world", "中文字", "the quick brown fox"] {
            let first = assign_shard_fixed(w.as_bytes(), n);
            for _ in 0..50 {
                assert_eq!(
                    assign_shard_fixed(w.as_bytes(), n),
                    first,
                    "分片分配必须确定性"
                );
            }
        }
    }
}

// tests/model_contract_test.rs
//
// 模型契约 v1 的验收测试（docs/v2输入输出标准规范技术方案.md §11）。
//
//   T-1 元信息往返         T-2 V1–V5 各一条违规模型必须报错（非静默）
//   T-3 v0 旧格式迁移      T-4 format_version 不兼容
//   T-5 向前兼容           T-6 输入侧计数上报
//   T-7 确定性固化（★ 最重要，跨独立进程 + 变 num_shards）
//   T-8 并行编码一致       T-8b special tokens round-trip（§8/C4 + Q7 端到端）
//   T-10 双剖面构建（#[ignore]，需跑 cargo）
//
// T-9（golden 不回归）由 tests/equiv_test.rs 承担，不在此重复。

use bbpe_rs::reader::{CorpusReader, CorpusStats};
use bbpe_rs::tokenizer::{BpeState, FORMAT_VERSION, LoadOptions, SamplingMeta, SaveOptions};
use bbpe_rs::{BBPETokenizer, BBPETrainer, RuntimeConfig, TrainerConfig};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// 与 track_a_test / equiv_test 同源的中英混合语料
const BASE_TEXT: &str = concat!(
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的",
    "能以人类智能相似的方式做出反应的智能机器。Python 是一种高级编程语言，",
    "以其简洁的语法和强大的生态系统而闻名。机器学习是人工智能的核心技术，",
    "深度学习是机器学习的重要分支。自然语言处理让计算机能够理解人类语言，",
    "大型语言模型在自然语言处理领域取得了巨大突破。The quick brown fox ",
    "jumps over the lazy dog. Machine learning models learn from large amounts ",
    "of data. Deep learning networks use multiple layers. 1234567890"
);

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mct_{tag}_{}_{}.{ext}",
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// 训练一个小模型，返回 (BpeState, 临时语料路径)
fn train_small(target_merges: usize, repeat: usize) -> BpeState {
    let corpus = tmp_path("corpus", "txt");
    // 逐行写入（BASE_TEXT 自身无换行），使 total_lines == repeat 可断言
    std::fs::write(&corpus, format!("{BASE_TEXT}\n").repeat(repeat)).unwrap();
    let reader = CorpusReader::Text {
        filepath: corpus.to_string_lossy().into_owned(),
    };
    let config = TrainerConfig::new(256 + target_merges);
    let mut trainer = BBPETrainer::new(config, None).unwrap();
    let stats = Arc::new(CorpusStats::default());
    trainer.train_with_stats(&reader, stats.clone()).unwrap();
    let meta = trainer.state.meta_mut();
    meta.profile = Some("v7".to_string());
    meta.sampling = Some(SamplingMeta {
        rate: 0.02,
        seed: 42,
        heldout_rate: 0.0005,
        heldout_seed: 7,
    });
    let st = trainer.state.clone();
    let _ = std::fs::remove_file(&corpus);
    st
}

// =========================================================
// T-1：元信息往返
// =========================================================
#[test]
fn t1_meta_roundtrip() {
    let st = train_small(40, 200);
    let path = tmp_path("t1", "json");
    st.save(path.to_str().unwrap()).unwrap();

    let loaded = BpeState::load(path.to_str().unwrap()).unwrap();
    let m = loaded.meta.as_ref().expect("meta 必须随模型往返");
    assert_eq!(m.profile.as_deref(), Some("v7"), "profile 标记错误");
    assert!(m.created_at.is_some(), "created_at 应自动补齐");
    assert!(m.bbpe_rs_version.is_some(), "bbpe_rs_version 应自动补齐");
    assert!(m.git_commit.is_some(), "git_commit 应自动补齐");

    let t = m.train.as_ref().expect("train 段由 trainer 写入");
    assert_eq!(t.actual_merges, 40);
    assert_eq!(t.end_reason, "vocab_size_reached");
    assert_eq!(t.min_frequency, 2);
    assert_eq!(t.max_chunk_bytes, 512);

    let c = m
        .corpus
        .as_ref()
        .expect("corpus 段由 train_with_stats 写入");
    assert_eq!(c.reader, "text");
    assert_eq!(c.total_lines, 200, "重复 200 行");
    assert_eq!(c.skipped_lines, 0, "Text reader 永不 skip");
    assert_eq!(c.replacement_chars, 0);

    let s = m.sampling.as_ref().expect("sampling 段由调用方写入");
    assert_eq!((s.rate, s.seed, s.heldout_seed), (0.02, 42, 7));

    assert_eq!(loaded.format_version, Some(FORMAT_VERSION));

    // --no-meta：供同配置重训逐字节比对（§7.6）
    let no_meta_path = tmp_path("t1_nometa", "json");
    st.save_with_options(
        no_meta_path.to_str().unwrap(),
        SaveOptions { no_meta: true },
    )
    .unwrap();
    let nm = BpeState::load(no_meta_path.to_str().unwrap()).unwrap();
    assert!(nm.meta.is_none(), "--no-meta 必须省略整个 meta 块");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&no_meta_path);
}

// =========================================================
// T-2：V1–V5 各一条违规模型必须报错（非静默）
// =========================================================
/// 断言违规模型被拒：`check_load = true` 时同时断言**加载**报错（T-2 的落点）。
///
/// 例外：`added_tokens` 与 vocab 的字节一致性（V5 后半条）在**加载**路径恒成立——
/// `BpeState::load` 会用 added_tokens 文本覆盖 vocab 中同 id 条目（这是设计如此，
/// 保证 special 可解码）。故该条只在 state 级（程序内 `validate()`）断言。
fn expect_validation_err(st: BpeState, tag: &str, check_load: bool) {
    match st.validate() {
        Err(bbpe_rs::BbpeError::ModelValidation(msg)) => {
            assert!(!msg.is_empty(), "{tag}: 错误信息不得为空")
        }
        Err(e) => panic!("{tag}: 期望 ModelValidation，实际 {e:?}"),
        Ok(()) => panic!("{tag}: 违规模型通过了 state 级校验"),
    }

    if check_load {
        let path = tmp_path(tag, "json");
        st.save(path.to_str().unwrap()).unwrap();
        let res = BpeState::load(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        match res {
            Err(bbpe_rs::BbpeError::ModelValidation(_)) => {}
            Err(e) => panic!("{tag}: 期望 ModelValidation，实际 {e:?}"),
            Ok(_) => panic!("{tag}: 违规模型被静默加载了"),
        }
    }
}

#[test]
fn t2_validation_rules() {
    // V1：删掉一个基础字节
    let mut st = train_small(20, 100);
    st.vocab.remove(&7);
    expect_validation_err(st, "t2_v1", true);

    // V2：merge 引用不在 vocab 中的 id
    let mut st = train_small(20, 100);
    st.merges.insert((30000, 30001), 30002);
    expect_validation_err(st, "t2_v2", true);

    // V3：new_id 不连续（把最后一条改成大号）
    let mut st = train_small(20, 100);
    let last = 256 + 20 - 1;
    let key = *st
        .merges
        .iter()
        .find(|(_, nid)| **nid == last)
        .map(|(k, _)| k)
        .unwrap();
    st.merges.insert(key, 9999);
    expect_validation_err(st, "t2_v3", true);

    // V4：pattern 无法编译
    let mut st = train_small(20, 100);
    st.pre_tokenizer_pattern = "(".to_string();
    expect_validation_err(st, "t2_v4", true);

    // V5a：special token 的 id 不在尾部（≤ 最大 new_id）
    let mut st = train_small(20, 100);
    st.special_tokens.insert("<|bad|>".to_string(), 260);
    st.vocab.insert(260, b"<|bad|>".to_vec());
    expect_validation_err(st, "t2_v5a", true);

    // V5b：special token 的字节与 vocab 中同 id 条目不一致
    //      （加载路径会被 load 覆盖修正，故只做 state 级断言，见 expect_validation_err 注释）
    let mut st = train_small(20, 100);
    st.special_tokens.insert("<|x|>".to_string(), 5000);
    st.vocab.insert(5000, b"something-else".to_vec());
    // m14：这条不一致在 **load 路径会被 added_tokens 覆盖而自愈**，此前仅靠注释
    // 记录，没有任何可执行信号。这里显式标注该事实，避免后人误以为加载会报错。
    expect_validation_err(st, "t2_v5b", false);
}

/// M1 回归：负 vocab id 必须被拒绝，而不是在 `id as usize` 处越界 panic。
///
/// 背景：`i32 as usize` 对 -1 得到 `usize::MAX` 量级的值，会在 `BBPETokenizer`
/// 的 O(1) 预计算数组构建处直接 panic。故修复在两层设防：解析期拒绝 +
/// validate() 拒绝（V1 扩展）。
#[test]
fn t2_negative_vocab_id_rejected() {
    // ① state 级：validate() 必须拒绝（覆盖程序内构造/修改后的 state）
    let mut st = train_small(20, 100);
    st.vocab.insert(-1, b"x".to_vec());
    match st.validate() {
        Err(bbpe_rs::BbpeError::ModelValidation(msg)) => {
            assert!(
                msg.contains("负 id"),
                "错误信息应明确指出是负 id 问题，实际: {msg}"
            );
        }
        other => panic!("负 id 必须被 validate() 拒绝，实际: {other:?}"),
    }

    // ② 加载级：即使绕过 validate，解析期也必须返回错误而非 panic
    let src = tmp_path("t2_neg_src", "json");
    train_small(20, 100).save(src.to_str().unwrap()).unwrap();
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&src).unwrap()).unwrap();
    v["vocab"]["-1"] = serde_json::json!([104]);
    let bad = tmp_path("t2_neg", "json");
    std::fs::write(&bad, serde_json::to_string(&v).unwrap()).unwrap();

    match BpeState::load(bad.to_str().unwrap()) {
        Err(bbpe_rs::BbpeError::InvalidModelFormat(msg)) => {
            assert!(msg.contains("-1"), "错误信息应含 offending id: {msg}")
        }
        Err(e) => panic!("期望 InvalidModelFormat，实际 {e:?}"),
        Ok(_) => panic!("含负 id 的模型被加载了（M1 回归）"),
    }
    // ③ 上层 BBPETokenizer（默认 validate=true）同样必须报错而非 panic
    assert!(
        BBPETokenizer::load(bad.to_str().unwrap()).is_err(),
        "BBPETokenizer 加载负 id 模型必须报错而非 panic"
    );

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&bad);
}

/// M2 回归：语料含 NUL 字节时，`(0,0)` 是**合法且高频**的 pair，
/// 不得被"哨兵 0"误判为"无候选"而提前终止。
///
/// 旧实现用 `best_pair_u64 == 0` 表达"无候选"，而 `pack_pair(0,0) == 0` ⟹
/// 实测 4000 个 NUL 会训练出 `merges=0` 且 `end_reason=min_frequency_terminated`
/// （终止原因还是错的：真实原因是哨兵冲突，不是频次不足）。
#[test]
fn t7_nul_corpus_pair_zero_zero_is_mergeable() {
    let corpus = tmp_path("t7_nul", "txt");
    // 4000 个 NUL ⟹ pair (0,0) 频次约 3992，远高于 min_frequency=2
    let mut data = vec![0u8; 4000];
    data.push(b'\n');
    std::fs::write(&corpus, &data).unwrap();

    let reader = CorpusReader::Text {
        filepath: corpus.to_string_lossy().into_owned(),
    };
    let config = TrainerConfig::new(256 + 4);
    let mut trainer = BBPETrainer::new(config, None).unwrap();
    trainer.train(&reader).unwrap();

    let m = trainer
        .state
        .meta
        .as_ref()
        .and_then(|x| x.train.clone())
        .expect("train 段必须存在");

    assert!(
        m.actual_merges > 0,
        "M2 回归：NUL 语料下 (0,0) 是合法高频 pair，必须能合并（旧实现 merges=0）"
    );
    assert_eq!(
        m.end_reason, "vocab_size_reached",
        "终止原因应为 vocab_size_reached；若为 min_frequency_terminated 说明哨兵冲突又回来了"
    );
    // 第一条 merge 必须就是 (0,0)
    let first = trainer
        .state
        .merges
        .iter()
        .find(|(_, nid)| **nid == 256)
        .map(|(k, _)| *k)
        .expect("第 0 次合并应为 (0,0)");
    assert_eq!(first, (0, 0), "(0,0) 必须是首个被合并的 pair");

    let _ = std::fs::remove_file(&corpus);
}

// =========================================================
// T-3：v0 旧格式迁移
// =========================================================
#[test]
fn t3_legacy_migration() {
    let st = train_small(20, 100);
    let v0_path = tmp_path("t3_v0", "json");
    // 手工构造 v0 文件：只有 vocab / merges（正是 tests_tokenizer.json 的形态）
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tmp_path_of(&st)).unwrap()).unwrap();
    let obj = value.as_object_mut().unwrap();
    obj.remove("format");
    obj.remove("format_version");
    obj.remove("pre_tokenizer_pattern");
    obj.remove("meta");
    std::fs::write(&v0_path, serde_json::to_string(&value).unwrap()).unwrap();

    // ① 默认严格：必须报错（Q2 静默 fallback 的修复）
    match BpeState::load(v0_path.to_str().unwrap()) {
        Err(bbpe_rs::BbpeError::MissingPreTokenizerPattern) => {}
        Err(e) => panic!("期望 MissingPreTokenizerPattern，实际 {e:?}"),
        Ok(_) => panic!("v0 模型被静默加载了（Q2 回归）"),
    }

    // ② allow_legacy：可加载（回退 DEFAULT_PATTERN），且 V1–V5 照常执行
    let legacy = BpeState::load_with_options(
        v0_path.to_str().unwrap(),
        LoadOptions {
            allow_legacy: true,
            validate: true,
        },
    )
    .expect("allow_legacy 应能加载 v0 模型");
    assert_eq!(
        legacy.format_version, None,
        "v0 的 format_version 应为 None"
    );

    // ③ 迁移：显式指定 pattern（不得猜）后成为 v1，且严格加载通过
    let v1_path = tmp_path("t3_v1", "json");
    let mut migrated = legacy.clone();
    migrated.pre_tokenizer_pattern = st.pre_tokenizer_pattern.clone();
    migrated.save(v1_path.to_str().unwrap()).unwrap();
    let reloaded = BpeState::load(v1_path.to_str().unwrap()).expect("迁移后应可严格加载");
    assert_eq!(reloaded.format_version, Some(FORMAT_VERSION));
    assert_eq!(reloaded.pre_tokenizer_pattern, st.pre_tokenizer_pattern);

    let _ = std::fs::remove_file(&v0_path);
    let _ = std::fs::remove_file(&v1_path);
}

/// 把 state 落到一个临时文件并返回其路径（供构造 v0 文件用）
fn tmp_path_of(st: &BpeState) -> std::path::PathBuf {
    let p = tmp_path("t3_src", "json");
    st.save(p.to_str().unwrap()).unwrap();
    p
}

// =========================================================
// T-4：format_version 不兼容
// =========================================================
#[test]
fn t4_unsupported_format_version() {
    let st = train_small(20, 100);
    let path = tmp_path("t4", "json");
    st.save(path.to_str().unwrap()).unwrap();
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    value["format_version"] = serde_json::json!(99);
    std::fs::write(&path, serde_json::to_string(&value).unwrap()).unwrap();

    match BpeState::load(path.to_str().unwrap()) {
        Err(bbpe_rs::BbpeError::UnsupportedFormatVersion(found, supported)) => {
            assert_eq!((found, supported), (99, FORMAT_VERSION));
        }
        Err(e) => panic!("期望 UnsupportedFormatVersion，实际 {e:?}"),
        Ok(_) => panic!("更高 format_version 被静默加载了"),
    }
    let _ = std::fs::remove_file(&path);
}

// =========================================================
// T-5：向前兼容 —— 旧加载器（忽略未知字段）能读 v1 模型
// =========================================================
#[derive(serde::Deserialize)]
struct LegacyModel {
    vocab: HashMap<String, Vec<u8>>,
    merges: HashMap<String, i32>,
}

#[test]
fn t5_forward_compatibility() {
    let st = train_small(20, 100);
    let path = tmp_path("t5", "json");
    st.save(path.to_str().unwrap()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let legacy: LegacyModel =
        serde_json::from_str(&text).expect("旧加载器（忽略未知字段）应能读 v1 模型");
    assert_eq!(legacy.vocab.len(), st.vocab.len());
    assert_eq!(legacy.merges.len(), st.merges.len());
    let _ = std::fs::remove_file(&path);
}

// =========================================================
// T-6：输入侧计数上报（§6.2 / §6.3 / §6.6）
// =========================================================
#[test]
fn t6_input_counters() {
    // ---- Text reader：非法 UTF-8 计入 replacement_chars，永不 skip ----
    let mut raw = Vec::new();
    raw.extend_from_slice(b"hello world hello\n");
    raw.extend_from_slice(b"\xff\xfe bad bytes\n"); // 2 个非法字节
    raw.extend_from_slice(b"\n"); // 空行：不计语料、不计 skip
    raw.extend_from_slice(b"hello world hello\n");
    let text_path = tmp_path("t6", "txt");
    std::fs::write(&text_path, &raw).unwrap();

    let reader = CorpusReader::Text {
        filepath: text_path.to_string_lossy().into_owned(),
    };
    let stats = Arc::new(CorpusStats::default());
    let lines: Vec<String> = reader
        .iter_lines_with_stats(stats.clone())
        .unwrap()
        .collect();
    assert_eq!(lines.len(), 3, "空行跳过，剩 3 行");
    let (total_lines, total_bytes, skipped, replaced) = stats.snapshot();
    assert_eq!(total_lines, 4, "物理行数含空行");
    assert_eq!(total_bytes, raw.len() as u64);
    assert_eq!(skipped, 0, "Text reader 永不 skip（§6.3）");
    assert_eq!(replaced, 2, "2 个非法字节 → 2 个 U+FFFD");

    // ---- JsonlStaticText：坏行 / 缺 text 字段计入 skipped_lines ----
    let jsonl = concat!(
        "{\"text\":\"hello world hello\"}\n",
        "{bad json\n",            // 解析失败
        "{\"other\":\"nope\"}\n", // text 缺失
        "\n",                     // 空行：不计
        "{\"text\":\"hello world hello\"}\n",
    );
    let jsonl_path = tmp_path("t6", "jsonl");
    std::fs::write(&jsonl_path, jsonl).unwrap();

    let reader = CorpusReader::JsonlStaticText {
        filepath: jsonl_path.to_string_lossy().into_owned(),
    };
    let stats = Arc::new(CorpusStats::default());
    let lines: Vec<String> = reader
        .iter_lines_with_stats(stats.clone())
        .unwrap()
        .collect();
    assert_eq!(lines.len(), 2);
    let (total_lines, _bytes, skipped, _replaced) = stats.snapshot();
    assert_eq!(total_lines, 5);
    assert_eq!(skipped, 2, "坏行 + 缺 text 字段各计 1（§6.3）");

    let _ = std::fs::remove_file(&text_path);
    let _ = std::fs::remove_file(&jsonl_path);
}

// =========================================================
// T-7：确定性固化（★ 跨独立进程 × 变 num_shards）
// =========================================================

/// 用给定 num_shards 训练，返回 merges 序列的规范化字符串（None = 默认线程数）
fn merges_digest(num_shards: Option<usize>) -> String {
    let corpus = tmp_path("t7_corpus", "txt");
    std::fs::write(&corpus, BASE_TEXT.repeat(120)).unwrap();
    let reader = CorpusReader::Text {
        filepath: corpus.to_string_lossy().into_owned(),
    };
    let config = TrainerConfig::new(256 + 60);
    let runtime = num_shards.map(|n| RuntimeConfig {
        num_shards: n,
        ..Default::default()
    });
    let mut trainer = BBPETrainer::new(config, runtime).unwrap();
    trainer.train(&reader).unwrap();

    let mut seq: Vec<(i32, (i32, i32))> = trainer
        .state
        .merges
        .iter()
        .map(|(&(a, b), &nid)| (nid, (a, b)))
        .collect();
    seq.sort_unstable_by_key(|e| e.0);

    // M3：除 merges 序列外，额外把 `--no-meta` 模型写到指定路径（若给出），
    // 供父进程做**文件字节级**比对。这是 `SaveOptions::no_meta` 承诺
    // "同配置重训可逐字节比对" 的唯一真正守护 —— 只比 merges 序列是
    // 比不出序列化键序随机的（见 M3）。
    if let Ok(p) = std::env::var("BBPE_T7_OUT") {
        trainer
            .save_with_options(&p, SaveOptions { no_meta: true })
            .unwrap();
    }

    let _ = std::fs::remove_file(&corpus);
    format!("{seq:?}")
}

#[test]
fn t7_determinism_across_processes_and_num_shards() {
    const TAG: &str = "T7DIGEST:";
    const TEST: &str = "t7_determinism_across_processes_and_num_shards";

    // 子进程模式：跑一次训练并把摘要打到 stdout
    if std::env::var("BBPE_T7_CHILD").is_ok() {
        let shards: Option<usize> = match std::env::var("BBPE_T7_SHARDS").unwrap().as_str() {
            "default" => None,
            other => Some(other.parse().unwrap()),
        };
        println!("{TAG}{}", merges_digest(shards));
        return;
    }

    let exe = std::env::current_exe().unwrap();
    let mut digests: Vec<(String, String, Vec<u8>)> = Vec::new();
    // 两个**独立进程**（覆盖 AHashMap 进程级随机种子，§6.5/C1）× num_shards ∈ {1,4,默认}
    for round in 0..2 {
        for shards in ["1", "4", "default"] {
            let model_out = tmp_path(&format!("t7_model_r{round}"), "json");
            let out = std::process::Command::new(&exe)
                .args(["--exact", TEST, "--nocapture"])
                .env("BBPE_T7_CHILD", "1")
                .env("BBPE_T7_SHARDS", shards)
                .env("BBPE_T7_OUT", &model_out)
                .output()
                .expect("子进程执行失败");
            assert!(out.status.success(), "子进程训练失败（shards={shards}）");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let digest = stdout
                .lines()
                .find_map(|l| l.trim().strip_prefix(TAG))
                .unwrap_or_else(|| panic!("子进程未输出摘要: {stdout}"))
                .to_string();
            let bytes = std::fs::read(&model_out)
                .unwrap_or_else(|e| panic!("子进程未产出 no_meta 模型: {e}"));
            let _ = std::fs::remove_file(&model_out);
            digests.push((format!("round{round}/shards={shards}"), digest, bytes));
        }
    }

    // ① 逻辑层：merges 序列一致
    let first = &digests[0].1;
    assert!(!first.is_empty());
    for (label, d, _) in &digests[1..] {
        assert_eq!(
            first, d,
            "确定性破坏：{} 与 {} 的 merges 序列不一致（需引入确定性 tie-break，见 §6.5/R3）",
            digests[0].0, label
        );
    }

    // ② M3：文件层 —— `no_meta` 模型必须**逐字节**一致。
    // 这是"同配置重训可逐字节比对"承诺的真正守护：只比 merges 序列看不出
    // 序列化键序随机（逻辑相同但文件字节不同，正是 M3 的失效模式）。
    let first_bytes = &digests[0].2;
    assert!(!first_bytes.is_empty(), "no_meta 模型不应为空");
    for (label, _, bytes) in &digests[1..] {
        assert_eq!(
            first_bytes, bytes,
            "M3 回归：{} 与 {} 的 no_meta 模型文件**字节不一致** \
             （序列化键序随机 ⟹ 逐字节比对承诺失效）",
            digests[0].0, label
        );
    }
}

// =========================================================
// T-8：并行编码与串行一致
// =========================================================
#[test]
fn t8_parallel_encode_matches_serial() {
    let st = train_small(40, 50);
    let path = tmp_path("t8", "json");
    st.save(path.to_str().unwrap()).unwrap();
    let tok = BBPETokenizer::load(path.to_str().unwrap()).unwrap();

    let text = BASE_TEXT.repeat(60); // 远超 5000 字节阈值 ⟹ 走 rayon 并行分支
    assert!(text.len() > 5000);

    let serial = tok.encode_ordinary(&text);
    let parallel = tok.encode_ordinary_parallel(&text, None);
    assert_eq!(serial, parallel, "并行编码与串行结果不一致（§8）");

    let enc_serial = tok.encode(&text, &std::collections::HashSet::new(), false);
    let enc_parallel = tok.encode_parallel(&text, &std::collections::HashSet::new(), false, None);
    assert_eq!(enc_serial.ids, enc_parallel.ids);
    assert_eq!(enc_serial.tokens, enc_parallel.tokens);

    // round-trip 契约（含 offsets 语义）
    let decoded = tok.decode(&serial).unwrap();
    assert_eq!(decoded, text, "decode(encode(t)) 必须无损（§8）");

    let _ = std::fs::remove_file(&path);
}

// =========================================================
// T-8b：含 special tokens 的 round-trip 契约（§8 / v1.1 C4，Q7 端到端守护）
// =========================================================
#[test]
fn t8b_special_tokens_roundtrip_and_boundary() {
    let specials = ["<|im_start|>", "<|im_end|>"];

    // ---- 边界（§6.4）：词表装不下 256 基础字节 + specials 时必须报错 ----
    // （否则 base_vocab_size < 256，special id 回卷覆盖基础字节，V1 被静默破坏）
    let mut bad = TrainerConfig::new(257);
    bad.special_tokens = vec!["<|im_start|>".to_string(), "<|im_end|>".to_string()];
    match BBPETrainer::new(bad, None) {
        Err(bbpe_rs::BbpeError::VocabSizeTooSmall(min)) => {
            assert_eq!(min, 258, "下界必须计入 special 坑位：256 + 2");
        }
        Err(e) => panic!("期望 VocabSizeTooSmall，实际 {e:?}"),
        Ok(_) => panic!("special 坑位未计入词表下界检查"),
    }

    // ---- 训练（specials 占尾部 2 个坑位）----
    let corpus = tmp_path("t8b", "txt");
    std::fs::write(&corpus, format!("{BASE_TEXT}\n").repeat(150)).unwrap();
    let reader = CorpusReader::Text {
        filepath: corpus.to_string_lossy().into_owned(),
    };
    let mut config = TrainerConfig::new(256 + 40 + specials.len());
    config.special_tokens = specials.iter().map(|s| s.to_string()).collect();
    let mut trainer = BBPETrainer::new(config, None).unwrap();
    trainer.train(&reader).unwrap();
    let _ = std::fs::remove_file(&corpus);

    let st = trainer.state.clone();
    let base = 256 + 40; // base_vocab_size = vocab_size - special 数
    let (id_start, id_end) = (
        st.special_tokens["<|im_start|>"],
        st.special_tokens["<|im_end|>"],
    );
    assert_eq!(
        (id_start, id_end),
        (base, base + 1),
        "special id 应在词表尾部（base_vocab_size + i）"
    );

    let path = tmp_path("t8b", "json");
    st.save(path.to_str().unwrap()).unwrap();
    // 严格加载默认执行 V1–V5 ⟹ 此处同时守护 V5 的三方一致性（Q7）
    let tok = BBPETokenizer::load(path.to_str().unwrap()).unwrap();

    // ---- round-trip（C4）：白名单内的 special 整体成词，decode 无损 ----
    let text = format!("<|im_start|>{BASE_TEXT}<|im_end|>");
    let allowed: HashSet<&str> = specials.iter().copied().collect();
    let enc = tok.encode(&text, &allowed, false);
    assert!(
        enc.ids.contains(&id_start) && enc.ids.contains(&id_end),
        "白名单内的 special 必须整体成词"
    );
    assert_eq!(
        tok.decode(&enc.ids).unwrap(),
        text,
        "decode(encode(t, allowed_special)) 必须无损（§8/C4）"
    );

    // ---- round-trip：白名单外的 special 被切碎成字节，decode 仍无损 ----
    let ordinary = tok.encode(&text, &HashSet::new(), false);
    assert!(
        !ordinary.ids.contains(&id_start) && !ordinary.ids.contains(&id_end),
        "未授权 special 不得整体成词（防注入）"
    );
    assert_eq!(
        tok.decode(&ordinary.ids).unwrap(),
        text,
        "切碎后的 round-trip 仍须无损"
    );

    // ---- 截断后 state 级 validate() 幂等（specials 由 truncate_to_k 重注入）----
    let mut cut = st.clone();
    cut.truncate_to_k(20);
    cut.validate()
        .expect("截断后的 state 必须仍通过 V1–V5（special 已重新注入 vocab）");

    let _ = std::fs::remove_file(&path);
}

// =========================================================
// C1 回归（2026-10-12 评审）：空 special token 必须被拒绝，
// 而不是"契约放行、encode 一跑就 panic"。
//
// 背景：AhoCorasick 的空 pattern 在每个字节位都匹配（含 UTF-8 字符内部的
// 延续字节位置）⟹ encode 的 `&text[start..end]` 在非字符边界切片直接 panic。
// 防线分三层：load 解析期拒绝（validate=false 时同样生效）→ validate() V6 →
// encode/encode_parallel 内 `start == end` 防御性跳过。
// =========================================================
#[test]
fn c1_empty_special_token_rejected() {
    // ① state 级：validate() 必须拒绝（覆盖程序内构造/修改后的 state）
    let mut st = train_small(20, 100);
    st.special_tokens.insert(String::new(), 5000);
    st.vocab.insert(5000, Vec::new());
    match st.validate() {
        Err(bbpe_rs::BbpeError::ModelValidation(msg)) => {
            assert!(
                msg.contains("V6") && msg.contains("空"),
                "错误信息应明确指出 V6 / 空 special token，实际: {msg}"
            );
        }
        other => panic!("空 special token 必须被 validate() 拒绝，实际: {other:?}"),
    }

    // ② 加载级：模型 JSON 的 added_tokens 含空串 ⟹ 解析期报错（先于 validate）
    let src = tmp_path("c1_src", "json");
    train_small(20, 100).save(src.to_str().unwrap()).unwrap();
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&src).unwrap()).unwrap();
    v["added_tokens"] = serde_json::json!({ "": 5000 });
    let bad = tmp_path("c1", "json");
    std::fs::write(&bad, serde_json::to_string(&v).unwrap()).unwrap();

    for validate in [true, false] {
        let opts = LoadOptions {
            allow_legacy: false,
            validate,
        };
        match BpeState::load_with_options(bad.to_str().unwrap(), opts) {
            Err(bbpe_rs::BbpeError::InvalidModelFormat(msg)) => {
                assert!(
                    msg.contains("空字符串"),
                    "错误信息应明确指出空 special token，实际: {msg}"
                );
            }
            Err(e) => panic!("validate={validate}: 期望 InvalidModelFormat，实际 {e:?}"),
            Ok(_) => panic!("validate={validate}: 含空 special token 的模型被静默加载了"),
        }
    }

    // ③ 上层 BBPETokenizer：报错而非 panic（C1 的崩点就在 encode 里）
    assert!(
        BBPETokenizer::load(bad.to_str().unwrap()).is_err(),
        "BBPETokenizer 加载含空 special token 的模型必须报错而非 panic"
    );

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&bad);
}

// =========================================================
// T-10：统一构建（较慢，默认忽略）
//
// 旧版 T-10 校验的是"v2 / v7 两个剖面各自编出哪些 bin"。剖面概念已删除
// （`track-a` 是空 feature，两剖面并无代码差异），现在**一次构建产出全部 bin**，
// 发布形态退化为打包项：完整包 = 全部 bin，v2 演示包 = 其中的子集。
// 本测试守护的就是这条性质：两个产物必须来自同一次构建，而非两条分支。
// =========================================================
#[test]
// 2026-10-07：保持 ignore，但**补上真正的执行场合** ——
// 此前它只被 `cargo test -- --ignored` 手工触发，等于"三支柱·支柱三（零剖面）"
// 这条发布关键路径**没有任何门禁**。现在 `scripts/release.sh` 的 ③ 步会跑它，
// 于是"一次构建产出全部 bin"在发布前是被验证的。
//
// 仍然 ignore 的原因：它用**全新的 target 目录**做 release 全量重建（含全部依赖），
// 耗时以分钟计。放进默认 `cargo test` 会让整个测试套件不可用 —— 这是"慢"，
// 不是"不该跑"。发布这种一次性场景正是它该跑的场合。
#[ignore = "需全新 target 目录做 release 全量重建（分钟级）；由 scripts/release.sh 自动执行，也可手工：cargo test --release -- --ignored"]
fn t10_unified_build() {
    // m15：① target 目录改为进程唯一 —— 原固定名 `bbpe_unified_target` 会让并行
    //       运行（debug/release、nextest）互相踩构建缓存，产生难解的构建失败；
    //     ② 用编译期展开的 `env!("CARGO")` 定位 cargo，不再依赖 PATH 中恰好有
    //       cargo（conda/自定义工具链环境下 PATH 里的可能是另一个版本）。
    let target = std::env::temp_dir().join(format!(
        "bbpe_unified_target_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let out = std::process::Command::new(env!("CARGO"))
        .args([
            "build",
            "--release",
            "--no-default-features",
            "--target-dir",
            target.to_str().unwrap(),
        ])
        .output()
        .expect("cargo 构建失败");
    assert!(
        out.status.success(),
        "统一构建失败:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bin_dir = target.join("release");

    // ① 统一入口 + 六个子命令
    for must in [
        "bbpe", "train", "sample", "sweep", "cut", "inspect", "compare",
    ] {
        assert!(bin_dir.join(must).exists(), "统一构建应产出 {must}");
    }
    // ①-b R4-27：`encode` / `decode` 是文档与 使用说明 直接引用的入口，
    // 但此前既不在本清单、也不在 release.sh 的 FULL_BINS 里 ⟹ 完整包不带它们。
    for must in ["encode", "decode"] {
        assert!(
            bin_dir.join(must).exists(),
            "应产出编解码入口 {must}（R4-27）"
        );
    }
    // ② v2 演示产物所需的 bin（完整产物的**子集**：训练 + 体检最小闭环）
    for must in ["train", "inspect"] {
        assert!(bin_dir.join(must).exists(), "v2 演示产物需要 {must}");
    }
    // ③ 过渡别名仍在（老脚本不至于立刻崩）
    for must in [
        "train_vocab",
        "sweep_vocab",
        "cut_vocab",
        "sample_corpus",
        "inspect_model",
        "compare_merges",
    ] {
        assert!(bin_dir.join(must).exists(), "过渡别名应保留 {must}");
    }
    // ④ 通用基准 / 进阶工具
    for must in ["bench", "bbpe_jsonl_bench", "run_real_data", "freeze_probe"] {
        assert!(bin_dir.join(must).exists(), "应产出 {must}");
    }
}

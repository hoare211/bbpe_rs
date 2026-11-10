// tests/review_20261101.rs
//
// 第五轮评审（docs/reviews/2026-11-01/2026-11-01-code-review.md）的回归测试。
//
// 本轮补齐第四轮 4 项部分修复（R4-1 / R4-4 / R4-7 / R4-17）的残留测试缺口，
// 并守护 R5-3（构造后篡改 special_tokens 的复查防线）与 R4-21（含 special 的
// cut 口径）。R4-17 的序列级断言在 src/tokenizer.rs 的 `#[cfg(test)]` 内
// （与 FSM 实现同文件）；R4-1 的预检判定纯函数单测在 src/cli/util.rs 内。
//
// 已覆盖：
//   R4-7  四种 reader 的 IO 故障注入（strict ⟹ Err；非严格 ⟹ 计数 + 终止）
//   R4-1  CLI 训练失败后输入文件（与显式 --sample-out 样本）逐字节存活
//   R4-21 含 special 的 cut：打印口径与词表真实项数一致
//   R5-3  new() 之后改写 special_tokens，train()/reset() 路径均被复查拒绝
//   R4-4  encode_ordinary_checked / encode_ordinary_parallel_checked 硬失败

use bbpe_rs::reader::{CorpusReader, CorpusStats};
use bbpe_rs::tokenizer::BpeState;
use bbpe_rs::{BBPETokenizer, BBPETrainer, BbpeError, TrainerConfig};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

fn bbpe() -> Command {
    Command::new(env!("CARGO_BIN_EXE_bbpe"))
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bbpe_r5_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("建临时目录失败");
    d
}

fn expect_err<T>(r: Result<T, BbpeError>, what: &str) -> BbpeError {
    match r {
        Ok(_) => panic!("期望失败但成功了：{what}"),
        Err(e) => e,
    }
}

/// 四种 reader 各构造一个（都指向 `path`）。动态 reader 用最小 schema。
fn all_readers(path: &str) -> Vec<CorpusReader> {
    vec![
        CorpusReader::Text {
            filepath: path.to_string(),
        },
        CorpusReader::JsonlStaticText {
            filepath: path.to_string(),
        },
        CorpusReader::JsonlStaticConversations {
            filepath: path.to_string(),
            sep: "\n".to_string(),
        },
        CorpusReader::JsonlDynamic {
            filepath: path.to_string(),
            schema_rules: vec![(vec!["text".to_string()], "str".to_string())],
        },
    ]
}

/// 写一份小而确定的 JSONL 语料（足够产生若干次合并）。
fn write_corpus(dir: &Path, name: &str, lines: usize) -> PathBuf {
    let mut s = String::new();
    for i in 0..lines {
        s.push_str(&format!(
            "{{\"text\": \"hello world hello world byte level bpe #{i}\"}}\n"
        ));
    }
    let p = dir.join(name);
    std::fs::write(&p, s).expect("写语料失败");
    p
}

// =========================================================
// R4-7：IO 故障注入（四种 reader 同口径）
// =========================================================
//
// 故障注入手法（确定性、无需 NFS / 坏块）：Linux 上 `File::open` 对**目录**
// 成功，而 `read_until` 返回 EISDIR ⟹ 首次读取即产生真实的 IO 错误。
// 目录路径由 tmpdir 自身充当，不依赖任何外部资源。
#[cfg(target_os = "linux")]
#[test]
fn r4_7_strict_iter_errors_and_stops_on_io() {
    let dir = tmpdir("io_strict");
    let path = dir.to_str().unwrap().to_string();

    for reader in all_readers(&path) {
        let kind = reader.kind();
        let stats = Arc::new(CorpusStats::default());
        let mut it = reader
            .iter_lines_checked(Some(stats.clone()))
            .unwrap_or_else(|e| panic!("{kind}: 目录可 open，迭代器构造不应失败: {e}"));

        let first = it
            .next()
            .unwrap_or_else(|| panic!("{kind}: 应产出 IO 错误项"));
        match first {
            Err(BbpeError::IoError(_)) => {}
            other => panic!("{kind}: strict 首项应为 Err(IoError)，实际 {other:?}"),
        }
        assert_eq!(stats.io_errors(), 1, "{kind}: io_errors 应计 1");
        assert!(it.next().is_none(), "{kind}: 首个 IO 错误后迭代必须终止");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(target_os = "linux")]
#[test]
fn r4_7_non_strict_iter_counts_only() {
    let dir = tmpdir("io_loose");
    let path = dir.to_str().unwrap().to_string();

    for reader in all_readers(&path) {
        let kind = reader.kind();
        let stats = Arc::new(CorpusStats::default());
        let lines: Vec<String> = reader
            .iter_lines_with_stats(stats.clone())
            .unwrap()
            .collect();
        assert!(lines.is_empty(), "{kind}: IO 错误后不应有任何产出");
        assert_eq!(stats.io_errors(), 1, "{kind}: io_errors 应计 1（而非 0）");
        // 非严格兼容旧行为：该错误行也计入 skipped（历史口径保留）
        let (_, _, skipped, _) = stats.snapshot();
        assert_eq!(skipped, 1, "{kind}: 非严格模式保留 skipped +1 的旧口径");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(target_os = "linux")]
#[test]
fn r4_7_dynamic_reader_training_fails_on_io() {
    let dir = tmpdir("io_train");
    let path = dir.to_str().unwrap().to_string();

    // 评审点名的直达路径：Python JsonlFileReader.dynamic ⟹ CorpusReader::JsonlDynamic
    for reader in all_readers(&path) {
        let kind = reader.kind();
        let mut trainer = BBPETrainer::new(TrainerConfig::new(300), None).expect("构造应成功");
        let err = expect_err(trainer.train(&reader), &format!("{kind}: 训练必须显式失败"));
        assert!(
            matches!(err, BbpeError::IoError(_)),
            "{kind}: 应为 IoError，实际: {err}"
        );
        assert!(!trainer.is_trained(), "{kind}: 失败后不应残留 merges");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================
// R4-1：训练失败后输入文件存活（CLI 集成）
// =========================================================
//
// OOM 预检依赖 MemAvailable 实测、测试内无法稳定触发（判定纯函数的边界
// 单测在 src/cli/util.rs）。这里用 `--format yaml` 让流程在**采样与预检之后**
// 确定性失败，验证同一条铁律：任何失败分支都不得删除输入文件。
#[test]
fn r4_1_input_and_sample_survive_failure() {
    let dir = tmpdir("survive");
    let corpus = write_corpus(&dir, "corpus.jsonl", 40);
    let sample = dir.join("sample.jsonl");
    let model = dir.join("vocab.json");
    let original = std::fs::read(&corpus).unwrap();

    let out = bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--format",
            "yaml", // 非法格式 ⟹ 阶段 3 失败（采样已完成）
            "--sample-rate",
            "0.5",
            "--sample-out",
            sample.to_str().unwrap(),
            "--vocab",
            "300",
            "--output",
            model.to_str().unwrap(),
        ])
        .output()
        .expect("运行 train 失败");
    assert!(!out.status.success(), "非法 --format 必须失败");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--format"), "错误应指明 --format: {stderr}");

    // 铁律断言：输入逐字节保留
    assert_eq!(
        std::fs::read(&corpus).expect("输入文件必须仍然存在"),
        original,
        "训练失败后原始语料必须逐字节保留"
    );
    // 显式 --sample-out 的样本不被清理（is_temp=false，无 RAII 守卫）
    assert!(
        sample.exists() && std::fs::metadata(&sample).unwrap().len() > 0,
        "显式 --sample-out 的采样语料必须在失败后仍然保留"
    );
    assert!(!model.exists(), "失败时不应产出模型");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn r4_1_input_preserved_on_failure_without_sampling() {
    // R4-1 P0 的原始形态：不采样时 train_corpus == 用户原始语料，
    // 旧实现在失败分支 remove_file(&train_corpus) 会直接删掉用户数据。
    let dir = tmpdir("survive_nosample");
    let corpus = write_corpus(&dir, "corpus.jsonl", 40);
    let model = dir.join("vocab.json");
    let original = std::fs::read(&corpus).unwrap();

    let out = bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--format",
            "yaml",
            "--vocab",
            "300",
            "--output",
            model.to_str().unwrap(),
        ])
        .output()
        .expect("运行 train 失败");
    assert!(!out.status.success(), "非法 --format 必须失败");
    assert_eq!(
        std::fs::read(&corpus).expect("输入文件必须仍然存在"),
        original,
        "不采样路径失败后原始语料必须逐字节保留"
    );
    assert!(!model.exists(), "失败时不应产出模型");
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================
// R4-21：含 special 的 cut 口径
// =========================================================
#[test]
fn r4_21_cut_vocab_size_with_special_matches_print() {
    let dir = tmpdir("cut_special");
    let corpus = write_corpus(&dir, "corpus.jsonl", 120);
    let src_model = dir.join("src.json");
    let cut_model = dir.join("cut.json");

    // 带两个 special 训练（CLI train 不支持 special，故走 API）
    let mut cfg = TrainerConfig::new(320);
    cfg.special_tokens = vec!["<|s|>".to_string(), "<|e|>".to_string()];
    let mut trainer = BBPETrainer::new(cfg, None).expect("构造应成功");
    let reader = CorpusReader::JsonlStaticText {
        filepath: corpus.to_str().unwrap().to_string(),
    };
    trainer.train(&reader).expect("训练应成功");
    trainer
        .save(src_model.to_str().unwrap())
        .expect("保存应成功");

    let out = bbpe()
        .args([
            "cut",
            "--model",
            src_model.to_str().unwrap(),
            "--output",
            cut_model.to_str().unwrap(),
            "--vocab",
            "280", // 目标 k = 24
        ])
        .output()
        .expect("运行 cut 失败");
    assert!(
        out.status.success(),
        "cut 失败: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    // 真实项数 = 256 基础 + k merges + 2 special（truncate_to_k 会重注入 special）
    let st = BpeState::load(cut_model.to_str().unwrap()).expect("截断模型必须可加载");
    let k = st.merges.len();
    assert!(k <= 24, "k 不得超过目标 24，实际 {k}");
    assert_eq!(
        st.vocab.len(),
        256 + k + 2,
        "词表项数必须 = 256 + k + n_special"
    );
    assert!(
        st.special_tokens.contains_key("<|s|>") && st.special_tokens.contains_key("<|e|>"),
        "截断不得丢失 special token"
    );
    // 打印的项数与真实项数一致（R4-21 的核心：不再自相矛盾）
    assert!(
        stdout.contains(&format!("词表 {} 项", st.vocab.len())),
        "打印项数必须与 st.vocab.len() 一致: {stdout}"
    );

    // sweep 导出路径（cut_model）同口径：打印拆分 256 + k + n_special，
    // 返回的 k 与截断模型的 merges 数一致
    let sweep_out = dir.join("sweep_export.json");
    let k2 = bbpe_rs::cli::cut::cut_model(
        src_model.to_str().unwrap(),
        sweep_out.to_str().unwrap(),
        280,
        false,
    )
    .expect("cut_model 应成功");
    let st2 = BpeState::load(sweep_out.to_str().unwrap()).expect("导出模型必须可加载");
    assert_eq!(k2, st2.merges.len());
    assert_eq!(st2.vocab.len(), 256 + k2 + 2);
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================
// R5-3：构造后篡改 special_tokens 的复查防线
// =========================================================
#[test]
fn r5_3_tampered_special_rejected_at_train() {
    let dir = tmpdir("mutate_special");
    let corpus = dir.join("corpus.txt");
    std::fs::write(&corpus, "hello world hello world\n").unwrap();
    let reader = CorpusReader::Text {
        filepath: corpus.to_str().unwrap().to_string(),
    };

    // 空串：new() 之后改写 ⟹ train 复查拒绝（与 R4-20 同一份校验实现）
    let mut cfg = TrainerConfig::new(300);
    cfg.special_tokens = vec!["<|s|>".to_string()];
    let mut trainer = BBPETrainer::new(cfg, None).expect("合法配置构造应成功");
    trainer.trainer_config.special_tokens = vec![String::new()];
    let err = expect_err(
        trainer.train(&reader),
        "空串 special 必须在 train 入口被拒绝",
    );
    assert!(
        err.to_string().contains("空"),
        "错误应说明空字符串，实际: {err}"
    );

    // 重复项：同样必须被复查拒绝
    let mut cfg = TrainerConfig::new(300);
    cfg.special_tokens = vec!["<|s|>".to_string()];
    let mut trainer = BBPETrainer::new(cfg, None).expect("合法配置构造应成功");
    trainer.trainer_config.special_tokens = vec!["<|s|>".to_string(), "<|s|>".to_string()];
    let err = expect_err(
        trainer.train(&reader),
        "重复 special 必须在 train 入口被拒绝",
    );
    assert!(
        err.to_string().contains("重复"),
        "错误应说明重复项，实际: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn r5_3_retrain_after_reset_still_rejected() {
    let dir = tmpdir("mutate_reset");
    let corpus = dir.join("corpus.txt");
    std::fs::write(&corpus, "hello world hello world byte level bpe\n").unwrap();
    let reader = CorpusReader::Text {
        filepath: corpus.to_str().unwrap().to_string(),
    };

    let mut cfg = TrainerConfig::new(300);
    cfg.special_tokens = vec!["<|s|>".to_string()];
    let mut trainer = BBPETrainer::new(cfg, None).expect("构造应成功");
    trainer.train(&reader).expect("首次训练应成功");

    // reset() 会按 trainer_config 重建 state ⟹ 篡改后的配置同样走复查
    trainer.trainer_config.special_tokens = vec![String::new()];
    trainer.reset();
    let err = expect_err(trainer.train(&reader), "reset 后非法 special 仍必须被拒绝");
    assert!(
        err.to_string().contains("空"),
        "错误应说明空字符串，实际: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================
// R4-4：ordinary 的 checked 变体（硬失败）
// =========================================================
//
// 夹具：pattern 只覆盖小写字母 ⟹ 含中文的文本必然预分词失败。
// 训练语料用纯小写字母（无空格），避免训练阶段自身出现 gap。
#[test]
fn r4_4_ordinary_checked_errors_on_pretokenize_failure() {
    let dir = tmpdir("ordinary_checked");
    let corpus = dir.join("corpus.txt");
    let mut s = String::new();
    for i in 0..60 {
        s.push_str(&format!("abcdefghijklmnopqrstuvwxyz{i:03}\n"));
    }
    // 去掉数字：pattern 只覆盖 [a-z]+，数字会引入 gap
    let s: String = s.chars().filter(|c| !c.is_ascii_digit()).collect();
    std::fs::write(&corpus, s).unwrap();
    let model = dir.join("vocab.json");

    let mut cfg = TrainerConfig::new(300);
    cfg.pre_tokenizer_pattern = "[a-z]+".to_string();
    let mut trainer = BBPETrainer::new(cfg, None).expect("构造应成功");
    let reader = CorpusReader::Text {
        filepath: corpus.to_str().unwrap().to_string(),
    };
    trainer.train(&reader).expect("纯小写语料训练应成功");
    trainer.save(model.to_str().unwrap()).expect("保存应成功");
    let tok = BBPETokenizer::load(model.to_str().unwrap()).expect("模型必须可加载");

    // 严格变体：预分词失败 ⟹ Err(PreTokenize)，而非空/不完整结果
    for r in [
        tok.encode_ordinary_checked("abc中文"),
        tok.encode_ordinary_parallel_checked("abc中文", Some(2)),
    ] {
        let err = expect_err(r, "未覆盖文本必须硬失败");
        assert!(
            matches!(err, BbpeError::PreTokenize(_)),
            "应为 PreTokenize，实际: {err}"
        );
    }

    // 对照：纯 ASCII 文本两条路径结果一致且非空
    let strict = tok
        .encode_ordinary_checked("abcabc")
        .expect("合法文本不应失败");
    let parallel = tok
        .encode_ordinary_parallel_checked("abcabc", Some(2))
        .expect("合法文本不应失败");
    assert!(!strict.is_empty());
    assert_eq!(strict, parallel, "串行/并行严格路径结果必须一致");

    // 非严格快速路径的既定语义（R4-4 文档化）：失败返回**不完整**结果
    let partial = tok.encode_ordinary("abc中文");
    assert!(!partial.is_empty(), "非严格路径应返回已覆盖部分（abc）");
    assert_eq!(
        tok.decode(&partial).unwrap(),
        "abc",
        "非严格路径的产出应为已覆盖前缀，不得混入未覆盖字节"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

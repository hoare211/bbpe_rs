// tests/equiv_test.rs
//
// 等价性回归测试：保证所有内存/性能优化后，BPE 的 merges 序列与"黄金基准"逐条一致。
//
// 生成/更新 golden：
//   UPDATE_GOLDEN=1 cargo test --test equiv_test --release
// 常规验证：
//   cargo test --test equiv_test --release

use bbpe_rs::{BBPETrainer, CorpusReader, TrainerConfig};

/// 丰富的中英数标点混合文本，包含重复词以触发有意义的 merges 与 tie-break。
const BASE_TEXT: &str = concat!(
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的",
    "能以人类智能相似的方式做出反应的智能机器。Python 是一种高级编程语言，",
    "以其简洁的语法和强大的生态系统而闻名。机器学习是人工智能的核心技术，",
    "深度学习是机器学习的重要分支。自然语言处理让计算机能够理解人类语言，",
    "大型语言模型在自然语言处理领域取得了巨大突破。The quick brown fox ",
    "jumps over the lazy dog. Machine learning models learn from large amounts ",
    "of data. Deep learning networks use multiple layers. 1234567890"
);

const REPEAT: usize = 300;
const TARGET_MERGES: usize = 80;
// GOLDEN_PATH 是**随仓库提交的黄金基准**，保持固定相对路径是合理的（它就是被比对的对象）
const GOLDEN_PATH: &str = "tests/golden_merges.txt";

/// m15：临时语料必须**进程唯一**。
///
/// 原实现把中间语料写在固定的 `tests/equiv_corpus_run.txt`，而并行 `cargo test`
/// （例如同时跑 debug 与 release、或用 nextest）会有多个进程同时读写并互相
/// `remove_file` ⟹ 训练中途文件消失，报出与等价性无关的 NotFound（典型 flaky）。
/// 这里改用「pid + 原子计数」的唯一名（track_a_test.rs / model_contract_test.rs
/// 已是此范式），并在结束时清理。
fn run_corpus_path() -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "bbpe_equiv_corpus_{}_{}.txt",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// 训练并把 merges 导出为【确定性字符串】：按 new_id 升序，每行 `id <- p0 , p1`。
/// new_id 编码了合并先后顺序 (256 + i)，因此逐行对比即等价于对比合并序列。
fn train_and_dump() -> String {
    let run_corpus = run_corpus_path();
    let corpus = BASE_TEXT.repeat(REPEAT);
    std::fs::write(&run_corpus, &corpus).unwrap();

    let reader = CorpusReader::Text {
        filepath: run_corpus.to_string_lossy().into_owned(),
    };
    let config = TrainerConfig::new(256 + TARGET_MERGES);
    let mut trainer = BBPETrainer::new(config, None).unwrap();
    trainer.train(&reader).unwrap();

    // 清理临时语料
    let _ = std::fs::remove_file(&run_corpus);

    // 确定性导出：按 new_id 排序
    let mut merges: Vec<_> = trainer.state.merges.iter().collect();
    merges.sort_by_key(|e| *e.1);

    let mut out = String::new();
    for e in &merges {
        out.push_str(&format!("{} <- {} , {}\n", e.1, (e.0).0, (e.0).1));
    }

    // 同时导出 vocab 前缀校验（确保字节内容一致）
    out.push_str("--- vocab ---\n");
    let limit = 256 + TARGET_MERGES as i32;
    let mut vocab: Vec<_> = trainer
        .state
        .vocab
        .iter()
        .filter(|e| *e.0 >= 256 && *e.0 < limit)
        .collect();
    vocab.sort_by_key(|e| *e.0);
    for e in &vocab {
        out.push_str(&format!("{} = {:?}\n", e.0, e.1));
    }

    out
}

#[test]
fn test_equiv_golden() {
    let actual = train_and_dump();

    if std::env::var("UPDATE_GOLDEN").is_ok() {
        std::fs::write(GOLDEN_PATH, &actual).unwrap();
        eprintln!("golden 已写入 {}", GOLDEN_PATH);
        return;
    }

    let golden = std::fs::read_to_string(GOLDEN_PATH).unwrap_or_else(|_| {
        panic!(
            "golden 文件不存在：{}。\n请运行：\n  UPDATE_GOLDEN=1 cargo test --test equiv_test --release",
            GOLDEN_PATH
        )
    });

    assert_eq!(
        actual, golden,
        "merges/vocab 与 golden 不一致 —— 等价性被破坏！\n\
         若是有意修改基准，请用 UPDATE_GOLDEN=1 重新生成。"
    );
}

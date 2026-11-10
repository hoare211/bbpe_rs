//! R6-1 回归：训练取消令牌。
//!
//! 覆盖四类不变量：
//! 1. 取消必须走 `Err`（不是 break），且不写 `meta.train`；
//! 2. probe 能把外部信号变成取消意图；
//! 3. 未触发取消时，训练结果与不带令牌的基线**逐位一致**（确定性）；
//! 4. 训练中途由其它线程置位也能中断。
//!
//! 这些用例不依赖 pyo3，因此两种 feature 变体都能跑。

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bbpe_rs::cancel::{CancelReason, CancelToken};
use bbpe_rs::error::BbpeError;
use bbpe_rs::reader::CorpusReader;
use bbpe_rs::trainer::{BBPETrainer, TrainerConfig};

static SEQ: AtomicU64 = AtomicU64::new(0);

fn write_corpus(body: &str) -> String {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "bbpe_cancel_{}_{}_{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        n
    ));
    fs::write(&path, body).unwrap();
    path.to_string_lossy().into_owned()
}

/// 足够多的重复行，让阶段 1/2/3 都真实跑起来，但整体仍在毫秒~百毫秒级。
fn corpus_text() -> String {
    let line = "中国语言文字处理与字节级 BPE 分词器训练 abcabc 中文中文 0123456789 ";
    line.repeat(4000)
}

fn new_trainer(vocab: usize) -> BBPETrainer {
    BBPETrainer::new(TrainerConfig::new(vocab), None).unwrap()
}

fn assert_cancelled(err: &BbpeError) {
    assert!(
        matches!(err, BbpeError::Cancelled { .. }),
        "期望 Cancelled，实际得到 {err:?}"
    );
}

#[test]
fn pre_cancelled_errors_and_writes_no_meta() {
    let path = write_corpus(&corpus_text());
    let reader = CorpusReader::Text {
        filepath: path.clone(),
    };
    let mut trainer = new_trainer(320);

    let token = CancelToken::new();
    token.cancel(CancelReason::User);
    let err = trainer
        .train_with_cancel(&reader, None, Some(&token))
        .unwrap_err();

    assert_cancelled(&err);
    let meta_written = trainer
        .state
        .meta
        .as_ref()
        .and_then(|m| m.train.as_ref())
        .is_some();
    assert!(
        !meta_written,
        "取消路径绝不能写 meta.train（否则会被当成正常完成的产物）"
    );
    assert!(!trainer.is_trained(), "取消不应产出任何 merges");
    let _ = fs::remove_file(path);
}

#[test]
fn probe_signal_triggers_cancel() {
    let path = write_corpus(&corpus_text());
    let reader = CorpusReader::Text {
        filepath: path.clone(),
    };
    let mut trainer = new_trainer(320);

    let probed = Arc::new(AtomicBool::new(false));
    let flag = probed.clone();
    let token = CancelToken::with_probe(
        Some(Box::new(move || {
            flag.store(true, Ordering::Relaxed);
            true
        })),
        std::time::Duration::ZERO,
    );

    let err = trainer
        .train_with_cancel(&reader, None, Some(&token))
        .unwrap_err();
    assert_cancelled(&err);
    assert!(probed.load(Ordering::Relaxed), "probe 必须被调用");
    assert_eq!(token.reason(), CancelReason::Signal);
    let _ = fs::remove_file(path);
}

#[test]
fn no_cancel_matches_baseline_exactly() {
    let text = corpus_text();
    let a = write_corpus(&text);
    let b = write_corpus(&text);

    let mut baseline = new_trainer(320);
    baseline
        .train(&CorpusReader::Text {
            filepath: a.clone(),
        })
        .unwrap();

    let mut with_token = new_trainer(320);
    let token = CancelToken::new();
    with_token
        .train_with_cancel(
            &CorpusReader::Text {
                filepath: b.clone(),
            },
            None,
            Some(&token),
        )
        .unwrap();

    let mut base_merges: Vec<_> = baseline.state.merges.iter().collect();
    let mut tok_merges: Vec<_> = with_token.state.merges.iter().collect();
    base_merges.sort();
    tok_merges.sort();
    assert_eq!(
        base_merges, tok_merges,
        "插入取消检查点不得改变训练输出（确定性 T-7）"
    );
    assert!(!with_token.state.merges.is_empty());
    let _ = fs::remove_file(a);
    let _ = fs::remove_file(b);
}

#[test]
fn cancel_from_other_thread_interrupts() {
    let path = write_corpus(&corpus_text().repeat(20));
    let reader = CorpusReader::Text {
        filepath: path.clone(),
    };
    let mut trainer = new_trainer(2000);

    let token = Arc::new(CancelToken::new());
    let t2 = token.clone();
    let handle = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(5));
        t2.cancel(CancelReason::Timeout);
    });

    let err = trainer
        .train_with_cancel(&reader, None, Some(&*token))
        .unwrap_err();
    handle.join().unwrap();
    assert_cancelled(&err);
    assert_eq!(token.reason(), CancelReason::Timeout);
    let _ = fs::remove_file(path);
}

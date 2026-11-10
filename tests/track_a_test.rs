// tests/track_a_test.rs
//
// 轨 A 工具链回归测试：把方案中"承重但此前仅靠手工一次性实验验证"的三个性质
// 固化为常驻断言（对应 code review 缺口 P6）：
//
//   T-A1 采样嵌套性 + held-out 互斥性 + held-out 与 rate 无关（F18 回归）
//   T-A2 BPE 贪心前缀性质：truncate_to_k(K_max→k) 的 merges 与直接训练
//        vocab_size=256+k 的 merges 逐条一致（方案 §19.1 已手工实测 100,000/100,000，
//        此处小规模常驻守护）
//   T-A3 truncate_to_k 边界行为：K 超出 K_max 钳制；截断后词表自洽闭合
//        （所有 merge 的 key/new_id 均存在于词表且 < 256+k'）
//
// 另含参数校验回归：heldout_rate > 0 且未提供 held-out writer 必须报错（P4）。

use bbpe_rs::sampling::{SampleParams, SampleStats, sample_lines};
use bbpe_rs::tokenizer::BpeState;
use bbpe_rs::{BBPETrainer, CorpusReader, TrainerConfig};
use std::collections::HashSet;
use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};

/// 临时语料名唯一化计数器：多个 #[test] 并行执行，若仅用 (target_merges, pid) 命名，
/// 同名文件会被并发测试互相 `remove_file` 删除 ⟹ 训练时 NotFound（flaky 的根源）。
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

// ==========================================
// T-A1：F18 嵌套性 / held-out 互斥性 / held-out 稳定性
// ==========================================

/// 用行集合表示一个输出（每行一个元素，去掉行尾换行以便比较）
fn lines_of(buf: &[u8]) -> HashSet<String> {
    String::from_utf8_lossy(buf)
        .lines()
        .map(|s| s.to_string())
        .collect()
}

/// m14：原先 `run_sample` **丢弃了 SampleStats**，于是"训练 + held-out + 未采样
/// == 全量（不丢行）"这条最基础的不变量从来没有被断言过 —— 采样若少读行，
/// 所有嵌套性/互斥性断言依然会通过（它们只比较集合内容，不检查总量）。
/// 现在把 stats 一并返回，由调用方断言总量守恒。
fn run_sample(rate: f64, heldout_rate: f64) -> (Vec<u8>, Vec<u8>, SampleStats) {
    // 200 行确定性语料
    let mut corpus = Vec::new();
    for i in 0..200 {
        corpus.extend_from_slice(format!("line-{:06}\n", i).as_bytes());
    }
    let mut reader = Cursor::new(corpus);
    let mut train = Vec::new();
    let mut held = Vec::new();
    let stats = sample_lines(
        &mut reader,
        &mut train,
        Some(&mut held),
        SampleParams {
            rate,
            seed: 42, // train seed
            heldout_rate,
            heldout_seed: 7,
            max_bytes: u64::MAX,
        },
    )
    .unwrap();
    (train, held, stats)
}

#[test]
fn test_sample_nesting_exclusivity_f18() {
    let (train_small, held_small, st_small) = run_sample(0.3, 0.1);
    let (train_large, held_large, st_large) = run_sample(0.6, 0.1);

    // m14：守恒断言 —— 每一行都恰被归入「训练 / held-out / 未采样」三者之一，
    // 且总量必须等于输入行数。这条能挡住"采样少读行"这类静默丢数据，
    // 而原有的嵌套/互斥断言是**挡不住**的（它们只比较集合内容）。
    const N: u64 = 200;
    for st in [&st_small, &st_large] {
        assert_eq!(st.total_lines, N, "每个输入行都必须被计数（无丢行）");
        assert!(
            st.train_lines + st.held_lines <= st.total_lines,
            "训练 + held-out 不得超过总行数"
        );
        // 未采样行 = 总数 - 训练 - held-out，三者之和恒等于总数（不变量）
        let unsampled = st.total_lines - st.train_lines - st.held_lines;
        assert_eq!(
            st.train_lines + st.held_lines + unsampled,
            N,
            "训练 + held-out + 未采样 必须 == 全量"
        );
        // 字节口径同理
        assert!(
            st.train_bytes + st.held_bytes <= st.total_bytes,
            "训练 + held-out 字节不得超过总字节"
        );
    }
    // 两个 rng 每行都推进 ⟹ 不同 rate 下 total 读数必然一致
    assert_eq!(st_small.total_lines, st_large.total_lines);
    assert_eq!(st_small.total_bytes, st_large.total_bytes);

    let t_small = lines_of(&train_small);
    let t_large = lines_of(&train_large);
    let h_small = lines_of(&held_small);
    let h_large = lines_of(&held_large);

    // ① 同 seed 下 rate₁ < rate₂ ⟹ train₁ ⊆ train₂（严格嵌套，F18）
    for l in &t_small {
        assert!(
            t_large.contains(l),
            "嵌套性破坏：{l} 在 rate=0.3 中却不在 rate=0.6 中"
        );
    }
    assert!(
        !t_small.is_empty(),
        "rate=0.3 应采到至少一行（期望 ~60 行，实际 {}）",
        t_small.len()
    );
    assert!(
        t_large.len() >= t_small.len(),
        "rate=0.6 ({}) 应不少于 rate=0.3 ({})",
        t_large.len(),
        t_small.len()
    );

    // ② 训练集与 held-out 严格不相交（互斥）
    for l in &t_small {
        assert!(
            !h_small.contains(l),
            "互斥性破坏：{l} 同时落入训练集与 held-out"
        );
    }
    for l in &t_large {
        assert!(
            !h_large.contains(l),
            "互斥性破坏：{l} 同时落入训练集与 held-out"
        );
    }

    // ③ held-out 集合与 rate 无关（同 heldout seed/rate）
    assert_eq!(h_small, h_large, "held-out 集合不应随训练 rate 变化（F18）");
    assert!(
        !h_small.is_empty(),
        "heldout_rate=0.1 应采到至少一行（期望 ~20 行）"
    );
}

#[test]
fn test_sample_determinism_same_seed() {
    // 同 (seed, rate) 两次运行结果逐字节一致（固定种子可复现）
    let (a_train, a_held, a_stats) = run_sample(0.4, 0.05);
    let (b_train, b_held, b_stats) = run_sample(0.4, 0.05);
    assert_eq!(a_train, b_train);
    assert_eq!(a_held, b_held);
    // m14：统计量同样必须逐字段一致（同种子 ⟹ 完全可复现）
    assert_eq!(a_stats, b_stats);
}

#[test]
fn test_sample_heldout_rate_without_writer_is_error() {
    // P4 回归：--heldout-rate > 0 但未提供输出 writer 必须报错（旧行为是静默丢行）
    let mut reader = Cursor::new(b"a\nb\nc\n".to_vec());
    let mut train = Vec::new();
    let err = sample_lines(
        &mut reader,
        &mut train,
        None,
        SampleParams {
            rate: 0.5,
            seed: 42,
            heldout_rate: 0.1,
            heldout_seed: 7,
            max_bytes: u64::MAX,
        },
    )
    .expect_err("heldout_rate>0 且无 writer 必须报错");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn test_sample_max_bytes_keeps_whole_lines() {
    // --max-bytes 截断不产生半行，且截断点与 rate 无关（嵌套性在截断下保持）
    let corpus_of = || {
        let mut c = Vec::new();
        for i in 0..100 {
            c.extend_from_slice(format!("line-{:04}\n", i).as_bytes());
        }
        c
    };
    // m14：修正字节数注释并补齐断言。
    // `"line-{:04}\n"` = "line-"(5) + 4 位序号(4) + 换行(1) = **10** 字节，
    // 原注释写成 9（漏了序号位宽）。
    const LINE_BYTES: u64 = 10;
    assert_eq!(
        format!("line-{:04}\n", 0).len() as u64,
        LINE_BYTES,
        "行字节数假设必须与实际一致"
    );

    let mut r1 = Cursor::new(corpus_of());
    let mut t1 = Vec::new();
    let s1 = sample_lines(
        &mut r1,
        &mut t1,
        None,
        SampleParams {
            rate: 0.3,
            seed: 42,
            heldout_rate: 0.0,
            heldout_seed: 7,
            max_bytes: 10 * LINE_BYTES,
        },
    )
    .unwrap();
    let l1 = lines_of(&t1);
    // ① 不产生半行
    for l in &l1 {
        assert!(l.starts_with("line-"), "截断产生了半行: {l:?}");
        assert!(
            l.len() as u64 == LINE_BYTES - 1,
            "行长度异常（可能是半行）: {l:?}"
        );
    }
    // ② m14：`starts_with` 挡不住"少读行"。用计数断言兜住：
    //    上限 100 字节 / 每行 10 字节 ⟹ 恰好读满 10 行（含触发截断点那一行）
    assert_eq!(s1.total_lines, 10, "截断应恰好读满 10 行");
    assert_eq!(s1.total_bytes, 10 * LINE_BYTES, "截断应按整行读取字节");

    // 截断点与 rate 无关 ⟹ 不同 rate 的截断行集合是同一前缀
    let mut r2 = Cursor::new(corpus_of());
    let mut t2 = Vec::new();
    let s2 = sample_lines(
        &mut r2,
        &mut t2,
        None,
        SampleParams {
            rate: 0.9,
            seed: 42,
            heldout_rate: 0.0,
            heldout_seed: 7,
            max_bytes: 10 * LINE_BYTES,
        },
    )
    .unwrap();
    // ③ m14：截断点只依赖累计字节 ⟹ 与 rate 无关，两次的总读数必须相同
    assert_eq!(s1.total_lines, s2.total_lines, "截断点不应随 rate 变化");
    assert_eq!(s1.total_bytes, s2.total_bytes);
    let l2 = lines_of(&t2);
    for l in &l1 {
        assert!(
            l2.contains(l),
            "截断下嵌套性破坏：{l} 在 rate=0.3 中却不在 rate=0.9 中"
        );
    }
}

// ==========================================
// T-A2/T-A3：贪心前缀性质 + truncate_to_k 边界
// ==========================================

/// 与 equiv_test 同源的中英混合语料（足够产生 ≥80 个 merges）。
const BASE_TEXT: &str = concat!(
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的",
    "能以人类智能相似的方式做出反应的智能机器。Python 是一种高级编程语言，",
    "以其简洁的语法和强大的生态系统而闻名。机器学习是人工智能的核心技术，",
    "深度学习是机器学习的重要分支。自然语言处理让计算机能够理解人类语言，",
    "大型语言模型在自然语言处理领域取得了巨大突破。The quick brown fox ",
    "jumps over the lazy dog. Machine learning models learn from large amounts ",
    "of data. Deep learning networks use multiple layers. 1234567890"
);

/// 训练并返回 (按 new_id 升序的 merges 序列, BpeState)。
/// merges 序列元素 = (new_id, (p0, p1))，即第 i 项是第 i 次合并。
/// 按 new_id 升序还原的 merges 序列（元素为 `(new_id, (p0, p1))`）。
type MergeSeq = Vec<(i32, (i32, i32))>;

fn train_and_seq(target_merges: usize, repeat: usize) -> (MergeSeq, BpeState) {
    let corpus_path = std::env::temp_dir().join(format!(
        "track_a_corpus_{}_{}_{}.txt",
        target_merges,
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&corpus_path, BASE_TEXT.repeat(repeat)).unwrap();

    let reader = CorpusReader::Text {
        filepath: corpus_path.to_string_lossy().into_owned(),
    };
    let config = TrainerConfig::new(256 + target_merges);
    let mut trainer = BBPETrainer::new(config, None).unwrap();
    trainer.train(&reader).unwrap();
    let _ = std::fs::remove_file(&corpus_path);

    let mut seq: Vec<(i32, (i32, i32))> = trainer
        .state
        .merges
        .iter()
        .map(|(&(a, b), &nid)| (nid, (a, b)))
        .collect();
    seq.sort_unstable_by_key(|e| e.0);
    (seq, trainer.state)
}

#[test]
fn test_cut_prefix_equivalence() {
    // 方案 §19.1 实测结论的小规模常驻复现：
    // truncate_to_k(80→40) 与直接训练 vocab=256+40 的 merges **逐条一致**。
    let (seq_full, mut state_full) = train_and_seq(80, 300);
    let (seq_short, _state_short) = train_and_seq(40, 300);

    assert_eq!(seq_full.len(), 80, "语料应支撑满 80 个 merges");
    assert_eq!(seq_short.len(), 40, "语料应支撑满 40 个 merges");

    let kept = state_full.truncate_to_k(40);
    assert_eq!(kept, 40);

    let mut seq_cut: Vec<(i32, (i32, i32))> = state_full
        .merges
        .iter()
        .map(|(&(a, b), &nid)| (nid, (a, b)))
        .collect();
    seq_cut.sort_unstable_by_key(|e| e.0);

    assert_eq!(
        seq_cut, seq_short,
        "贪心前缀性质破坏：cut(80→40) 与直接训练 40 的 merges 不一致"
    );
}

#[test]
fn test_truncate_clamp_and_closure() {
    // T-A3a：k > K_max 时钳制到 K_max；k == K_max / k < K_max 正常
    let (seq, mut state) = train_and_seq(40, 300);
    let k_max = seq.len();
    assert_eq!(k_max, 40);

    let kept = state.truncate_to_k(1000);
    assert_eq!(kept, k_max, "k 超出 K_max 必须钳制到 K_max");
    assert_eq!(state.merges.len(), k_max);
    assert_eq!(state.vocab.len(), 256 + k_max);

    // T-A3b：截断后词表自洽闭合——每个 merge 的 key (a,b) 与 new_id 都在词表内，
    // 且 id < 256 + k'（由 new_id = 256 + i 的构造保证）
    let kept = state.truncate_to_k(17);
    assert_eq!(kept, 17);
    let limit = 256 + 17;
    for (&(a, b), &nid) in &state.merges {
        assert!(nid < limit, "new_id {nid} 越过截断线 {limit}");
        assert!(
            a < limit && b < limit,
            "merge key ({a},{b}) 越过截断线 {limit}"
        );
        assert!(
            state.vocab.contains_key(&a),
            "merge 输入 {a} 不在截断后词表中"
        );
        assert!(
            state.vocab.contains_key(&b),
            "merge 输入 {b} 不在截断后词表中"
        );
        assert!(
            state.vocab.contains_key(&nid),
            "merge 产物 {nid} 不在截断后词表中"
        );
    }
    for &id in state.vocab.keys() {
        assert!(id < limit, "词表残留越界 id {id}");
    }
}

#[test]
fn test_bpe_state_load_rejects_corrupt_model() {
    // P5 回归：坏模型文件必须报错而非静默吞掉（unwrap_or(0) 旧行为会把
    // 损坏模型伪装成合法 merges ((0,0)->v)，污染回读逻辑）
    let dir = std::env::temp_dir();
    let path = dir.join(format!("bad_model_{}.json", std::process::id()));

    // ① merges key 不是 "p0,p1" 整数对
    std::fs::write(
        &path,
        br#"{"vocab": {"256": [104,105]}, "merges": {"not_a_pair": 256}}"#,
    )
    .unwrap();
    assert!(
        BpeState::load(path.to_str().unwrap()).is_err(),
        "坏 merges key 必须报错"
    );

    // ② vocab key 不是整数
    std::fs::write(&path, br#"{"vocab": {"abc": [104]}, "merges": {}}"#).unwrap();
    assert!(
        BpeState::load(path.to_str().unwrap()).is_err(),
        "坏 vocab key 必须报错"
    );

    let _ = std::fs::remove_file(&path);
}

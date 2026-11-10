// tests/cli_test.rs
//
// 统一 CLI（`bbpe <子命令>`）的端到端冒烟，守护"三支柱"：
//
//   ① 单一入口：不传 `--sample-rate` 即全量精确训练（meta.sampling 缺省）；
//      传了就**真的采样**，且出身信息由程序据实写入（不靠人工转录）。
//   ② 单位统一：train / sweep / cut 三处的词表规模都是**总词表**（256 + K）。
//   ③ 一次构建：所有子命令由同一个 `bbpe` 二进制提供。
//
// 说明：用 `env!("CARGO_BIN_EXE_bbpe")` 直接跑构建产物，不依赖 PATH。

use bbpe_rs::tokenizer::BpeState;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn bbpe() -> Command {
    Command::new(env!("CARGO_BIN_EXE_bbpe"))
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bbpe_cli_test_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("建临时目录失败");
    d
}

/// 中英混合语料：行数与多样性都足够产生几百次合并
fn write_corpus(dir: &Path, lines: usize) -> PathBuf {
    let phrases = [
        "人工智能是计算机科学的一个分支",
        "机器学习需要大量高质量语料",
        "字节级 BPE 不依赖任何词典",
        "预分词 pattern 决定统计口径",
        "词表规模与压缩率的权衡",
        "held-out 评估必须与训练集互斥",
        "greedy prefix property 让选 K 免于重训",
        "deterministic training 是工程底线",
        "sampling rate 决定头部 pair 的精度",
        "fertility 越低说明 token 覆盖越差",
    ];
    let p = dir.join("corpus.jsonl");
    let mut s = String::new();
    // 用确定性伪随机（splitmix64 风格）拼句子，保证每次测试语料一致
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    for i in 0..lines {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let mut line = String::new();
        for j in 0..6 {
            line.push_str(phrases[(x.wrapping_add(j * 7) as usize) % phrases.len()]);
        }
        // 行尾加序号：保证每行内容唯一，从而可用"内容交集"验证训练/held-out 互斥
        // （真实语料允许重复行，那时互斥性是**按行位置**成立的，不能按内容比）
        s.push_str(&format!("{{\"text\": \"{line} #{i}\"}}\n"));
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    }
    std::fs::write(&p, s).expect("写语料失败");
    p
}

fn read_lines(p: &Path) -> Vec<String> {
    std::fs::read_to_string(p)
        .expect("读文件失败")
        .lines()
        .map(|l| l.to_string())
        .collect()
}

/// ① 全量训练：不采样 ⟹ meta.sampling 必须缺省（"全量"要能一眼看出来）
#[test]
fn t1_full_training_writes_no_sampling_meta() {
    let dir = tmpdir("full");
    let corpus = write_corpus(&dir, 900);
    let model = dir.join("vocab.json");

    let out = bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--format",
            "jsonl",
            "--vocab",
            "520",
            "--output",
            model.to_str().unwrap(),
        ])
        .output()
        .expect("运行 train 失败");
    assert!(
        out.status.success(),
        "train 失败: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let st = BpeState::load(model.to_str().unwrap()).expect("加载模型失败");
    assert_eq!(st.vocab.len(), 520, "总词表口径：--vocab 520 ⟹ 520 项");
    assert!(
        st.meta.as_ref().and_then(|m| m.sampling.clone()).is_none(),
        "全量训练不应写 sampling"
    );
    let c = st.meta.as_ref().unwrap().corpus.clone().unwrap();
    assert_eq!(c.reader, "jsonl:text");
    assert_eq!(c.total_lines, 900);

    let _ = std::fs::remove_dir_all(&dir);
}

/// ② 采样训练：出身信息由程序据实写入；训练集与 held-out 严格互斥
#[test]
fn t2_sampled_training_writes_exclusive_meta() {
    let dir = tmpdir("sampled");
    let corpus = write_corpus(&dir, 900);
    let model = dir.join("vocab_s.json");
    let sample = dir.join("sample.jsonl");
    let held = dir.join("heldout.jsonl");

    let out = bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--format",
            "jsonl",
            "--vocab",
            "520",
            "--output",
            model.to_str().unwrap(),
            "--sample-rate",
            "0.5",
            "--sample-seed",
            "42",
            "--heldout-rate",
            "0.05",
            "--heldout-seed",
            "43",
            "--heldout-out",
            held.to_str().unwrap(),
            "--sample-out",
            sample.to_str().unwrap(),
        ])
        .output()
        .expect("运行 train 失败");
    assert!(
        out.status.success(),
        "train(采样) 失败: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // 出身信息 = 事实，而非人工转录
    let st = BpeState::load(model.to_str().unwrap()).expect("加载模型失败");
    let s = st
        .meta
        .as_ref()
        .and_then(|m| m.sampling.clone())
        .expect("采样训练必须写 sampling");
    assert_eq!(s.rate, 0.5);
    assert_eq!(s.seed, 42);
    assert_eq!(s.heldout_rate, 0.05);
    assert_eq!(s.heldout_seed, 43);

    // 采样行数应显著少于全量，且 held-out 非空
    let train_lines = read_lines(&sample);
    let held_lines = read_lines(&held);
    assert!(
        train_lines.len() > 100 && train_lines.len() < 900,
        "采样行数异常: {}",
        train_lines.len()
    );
    assert!(!held_lines.is_empty(), "held-out 不应为空");

    // 互斥性（sample_lines 的既有保证，在 train 内建采样后仍必须成立）
    let train_set: HashSet<&String> = train_lines.iter().collect();
    assert!(
        held_lines.iter().all(|l| !train_set.contains(l)),
        "held-out 与训练集必须严格互斥"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// ③ 单位统一：cut 的 --vocab 与 train 同口径（总词表），不再需要心算 256+K
#[test]
fn t3_cut_shares_unit_with_train() {
    let dir = tmpdir("cut");
    let corpus = write_corpus(&dir, 900);
    let model = dir.join("vocab.json");
    let cut = dir.join("vocab_300.json");

    let out = bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--vocab",
            "520",
            "--output",
            model.to_str().unwrap(),
        ])
        .output()
        .expect("运行 train 失败");
    assert!(out.status.success());
    let src = BpeState::load(model.to_str().unwrap()).unwrap();
    assert!(
        src.merges.len() >= 44,
        "源模型 merges 太少，无法验证截断: {}",
        src.merges.len()
    );

    let out = bbpe()
        .args([
            "cut",
            "--model",
            model.to_str().unwrap(),
            "--vocab",
            "300",
            "--output",
            cut.to_str().unwrap(),
        ])
        .output()
        .expect("运行 cut 失败");
    assert!(
        out.status.success(),
        "cut 失败: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let st = BpeState::load(cut.to_str().unwrap()).unwrap();
    assert_eq!(
        st.vocab.len(),
        300,
        "cut --vocab 300 ⟹ 词表 300 项（总词表口径）"
    );
    assert_eq!(st.merges.len(), 44, "300 - 256 = 44 merges");
    // 出身信息必须继承
    assert!(st.meta.is_some(), "cut 应继承源模型的 meta");

    let _ = std::fs::remove_dir_all(&dir);
}

/// ④ sweep：总词表口径 + `--export-at` 直接产出模型（免手算、免第二次命令）
#[test]
fn t4_sweep_uses_total_vocab_and_exports() {
    let dir = tmpdir("sweep");
    let corpus = write_corpus(&dir, 900);
    let model = dir.join("vocab.json");

    let out = bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--vocab",
            "520",
            "--output",
            model.to_str().unwrap(),
        ])
        .output()
        .expect("运行 train 失败");
    assert!(out.status.success());

    let out = bbpe()
        .args([
            "sweep",
            "--model",
            model.to_str().unwrap(),
            "--held-out",
            corpus.to_str().unwrap(),
            "--vocabs",
            "300,400,520",
            "--export-at",
            "300",
        ])
        .output()
        .expect("运行 sweep 失败");
    assert!(
        out.status.success(),
        "sweep 失败: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let exported = dir.join("vocab_300.json");
    assert!(exported.exists(), "--export-at 300 应产出 vocab_300.json");
    let st = BpeState::load(exported.to_str().unwrap()).unwrap();
    assert_eq!(st.vocab.len(), 300, "导出的模型也必须是总词表口径");

    let _ = std::fs::remove_dir_all(&dir);
}

/// ⑤ 参数校验：held-out 组合非法时必须报错，而不是静默丢行
#[test]
fn t5_invalid_heldout_combo_errors() {
    let dir = tmpdir("badargs");
    let corpus = write_corpus(&dir, 100);

    let out = bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--vocab",
            "300",
            "--output",
            dir.join("m.json").to_str().unwrap(),
            "--heldout-rate",
            "0.05",
        ])
        .output()
        .expect("运行 train 失败");
    assert!(!out.status.success(), "缺 --heldout-out 时必须失败");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("heldout-out"), "错误信息应指明缺什么: {err}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// ⑥ inspect --validate-only 的退出码契约（CI 用）：0 = 通过
#[test]
fn t6_inspect_validate_only_exit_code() {
    let dir = tmpdir("inspect");
    let corpus = write_corpus(&dir, 900);
    let model = dir.join("vocab.json");

    bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--vocab",
            "400",
            "--output",
            model.to_str().unwrap(),
        ])
        .output()
        .expect("运行 train 失败");

    let st = bbpe()
        .args(["inspect", model.to_str().unwrap(), "--validate-only"])
        .status()
        .expect("运行 inspect 失败");
    assert!(st.success(), "自洽性校验应通过（退出码 0）");

    let _ = std::fs::remove_dir_all(&dir);
}

/// ⑦ B2：CLI 的"推论 7 判据"必须与模型自述的 `end_reason` 一致，不能反推。
///
/// 回归背景：`train` 曾用 `actual_merges >= target_merges` **反推**终止原因，
/// 而"提前终止"在 M2 之后有三种成因。于是语料合并空间耗尽
/// （`end_reason = no_more_pairs`）时，CLI 打印的是"min_frequency=2 才是终止
/// 原因"并给出**方向错误**的处置建议（下调 K / 增大样本；真实原因与 m 无关）。
/// 这与 A3 同类：两处描述同一件事，修了一处漏了兄弟。
#[test]
fn t7_inference_7_reads_real_end_reason() {
    let dir = tmpdir("endreason");
    // 低多样性语料（去重后仅 9 个词）⟹ 必然在目标词表之前耗尽合并空间。
    // 实测依据：同语料在 vocab=400 下 32 次合并后即 no_more_pairs。
    let corpus = dir.join("small.jsonl");
    let mut s = String::new();
    for _ in 0..300 {
        s.push_str("{\"text\": \"the quick brown fox jumps over the lazy dog\"}\n");
    }
    std::fs::write(&corpus, s).expect("写语料失败");
    let model = dir.join("vocab.json");

    let out = bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--format",
            "jsonl",
            "--vocab",
            "300",
            "--output",
            model.to_str().unwrap(),
        ])
        .output()
        .expect("运行 train 失败");
    assert!(
        out.status.success(),
        "train 失败: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();

    // 前提：该语料的真实终止原因确实是 no_more_pairs
    let st = BpeState::load(model.to_str().unwrap()).expect("加载模型失败");
    let reason = st
        .meta
        .as_ref()
        .and_then(|m| m.train.as_ref())
        .map(|t| t.end_reason.clone())
        .expect("训练必须写 meta.train 段");
    assert_eq!(
        reason, "no_more_pairs",
        "本测试的前提：低多样性小语料应耗尽合并空间（否则换语料）"
    );

    // 判据不得把终止原因归给 min_frequency（与模型自述矛盾）
    assert!(
        !stdout.contains("才是终止原因"),
        "no_more_pairs 时不得输出「min_frequency 才是终止原因」：\n{stdout}"
    );
    assert!(
        stdout.contains("no_more_pairs"),
        "CLI 应如实报出 end_reason：\n{stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// ⑧ B2 的另一支：`min_frequency` 早停时，判据必须报 min_frequency。
///
/// 与 ⑦ 合成"三种成因各有自己的文案"的完整守护 —— 只测 ⑦ 的话，
/// 把三个分支合并成一句"反正都是 min_frequency"也能通过。
#[test]
fn t8_early_stop_reports_min_frequency() {
    let dir = tmpdir("endreason_mf");
    let corpus = dir.join("small.jsonl");
    let mut s = String::new();
    for _ in 0..300 {
        s.push_str("{\"text\": \"the quick brown fox jumps over the lazy dog\"}\n");
    }
    std::fs::write(&corpus, s).expect("写语料失败");
    let model = dir.join("vocab.json");

    // min_frequency 高于任何 pair 的频次 ⟹ 最高频 pair 也达不到阈值
    let out = bbpe()
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--format",
            "jsonl",
            "--vocab",
            "300",
            "--min-frequency",
            "100000",
            "--output",
            model.to_str().unwrap(),
        ])
        .output()
        .expect("运行 train 失败");
    assert!(
        out.status.success(),
        "train 失败: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();

    let st = BpeState::load(model.to_str().unwrap()).expect("加载模型失败");
    let reason = st
        .meta
        .as_ref()
        .and_then(|m| m.train.as_ref())
        .map(|t| t.end_reason.clone())
        .expect("训练必须写 meta.train 段");
    assert_eq!(
        reason, "min_frequency_terminated",
        "min_frequency 高于一切 pair 频次 ⟹ 必须是 min_frequency_terminated"
    );
    assert!(
        stdout.contains("才是终止原因"),
        "min_frequency 早停时不得改报 no_more_pairs：\n{stdout}"
    );
    assert!(
        !stdout.contains("no_more_pairs"),
        "本分支不应出现 no_more_pairs 文案：\n{stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

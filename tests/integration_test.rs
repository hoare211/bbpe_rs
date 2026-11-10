// tests/integration_test.rs
//
// m15：原实现把语料与模型写在**当前工作目录**的固定名（`tests_corpus.txt` /
// `tests_tokenizer.json`），且用 `Once` + "文件存在就复用"做缓存守卫。两个问题：
//
//   ① 并行测试 / 多 profile（debug + release）会争抢同一批文件；
//   ② "仅查存在性"的守卫意味着**陈旧产物会被静默复用** —— 换了引擎版本后，
//      测试仍在拿旧模型验证，于是"通过"并不代表当前代码正确。
//
// 现在：产物放在本进程专属的临时目录，并**每次重新训练**（训练规模极小，
// 耗时可忽略），彻底消除跨运行污染与陈旧产物复用。

use bbpe_rs::{BBPETokenizer, BBPETrainer, CorpusReader, TrainerConfig};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Once, OnceLock};

static INIT: Once = Once::new();
/// 模型路径只初始化一次（`OnceLock` 保证并发调用者等待而非并发写同一文件）。
static MODEL_PATH: OnceLock<String> = OnceLock::new();

/// 本进程专属的临时目录（含 pid），避免与其它测试进程互相干扰。
/// 目录本身是全新的 ⟹ 不存在"陈旧产物被复用"的问题。
fn workdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("bbpe_integration_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn setup_test_environment() -> String {
    MODEL_PATH
        .get_or_init(|| {
            INIT.call_once(|| {
                let _ = tracing_subscriber::fmt().with_target(false).try_init();
            });

            let dir = workdir();
            let corpus_path = dir.join("corpus.txt");
            let model_path = dir.join("tokenizer.json");

            // m15：每次进程重新开始训练 —— 验证的必然是当前代码产出的模型，
            // 不会像"文件存在就复用"那样拿旧版本模型跑出假绿。
            let mut f = File::create(&corpus_path).unwrap();
            let text = "Hello 世界! 测试无损还原, 12345 罕见字龘。\n";
            for _ in 0..500 {
                f.write_all(text.as_bytes()).unwrap();
            }
            drop(f);

            let config = TrainerConfig::new(256 + 20);
            let mut trainer = BBPETrainer::new(config, None).unwrap();

            let reader = CorpusReader::Text {
                filepath: corpus_path.to_string_lossy().into_owned(),
            };
            trainer.train(&reader).unwrap();

            trainer.save(model_path.to_str().unwrap()).unwrap();
            model_path.to_string_lossy().into_owned()
        })
        .clone()
}

#[test]
fn test_round_trip_english_and_chinese() {
    let model_path = setup_test_environment();
    let tokenizer = BBPETokenizer::load(&model_path).unwrap();

    let text = "Hello 世界! 测试无损还原, 12345 罕见字龘。";
    let ids = tokenizer.encode_ordinary(text);
    let decoded = tokenizer.decode(&ids).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_empty_string() {
    let model_path = setup_test_environment();
    let tokenizer = BBPETokenizer::load(&model_path).unwrap();

    let ids = tokenizer.encode_ordinary("");
    assert!(ids.is_empty());

    let decoded = tokenizer.decode(&ids).unwrap();
    assert_eq!(decoded, "");
}

#[test]
fn test_compression_logic() {
    let model_path = setup_test_environment();
    let tokenizer = BBPETokenizer::load(&model_path).unwrap();

    let text = "测试无损还原";
    let raw_len = text.len();
    let ids = tokenizer.encode_ordinary(text);

    // 如果 BPE 训练成功，分词后的 ID 数组长度应该小于原生 utf-8 字节长度
    assert!(ids.len() < raw_len, "BPE 没有起到压缩效果");
}

#[test]
fn test_rust_specific_error_handling() {
    // 测试加载不存在的文件
    let res = BBPETokenizer::load("non_existent_file.json");
    assert!(res.is_err());

    // 测试 Decode 未知的 ID
    let model_path = setup_test_environment();
    let tokenizer = BBPETokenizer::load(&model_path).unwrap();
    let res = tokenizer.decode(&[999999]); // 肯定不在词表内
    assert!(res.is_err());
}

/// m2 回归：串行 `encode`（走 encode_normal_chunk → bpe_merge_core）与
/// `encode_ordinary`（裸 ID 路径）必须在 ids 上一致，且 tokens/offsets 与 ids
/// 长度对齐。三份合并循环合并成一份后，这条能挡住"某条路径漏改"。
#[test]
fn test_encode_paths_agree() {
    let model_path = setup_test_environment();
    let tokenizer = BBPETokenizer::load(&model_path).unwrap();

    let text = "Hello 世界! 测试无损还原, 12345 罕见字龘。混合 mixed content 123.";
    let ordinary = tokenizer.encode_ordinary(text);
    let full = tokenizer.encode(text, &std::collections::HashSet::new(), false);

    assert_eq!(ordinary, full.ids, "裸 ID 路径与全记录路径的 ids 必须一致");
    assert_eq!(
        full.tokens.len(),
        full.ids.len(),
        "tokens 与 ids 长度必须对齐"
    );
    assert_eq!(
        full.offsets.len(),
        full.ids.len(),
        "offsets 与 ids 长度必须对齐"
    );

    // offsets 必须单调不重叠且落在文本范围内。
    // 注意：字节级 BPE 的 token 可能**跨越 UTF-8 字符中间**，所以这里只能按
    // 字节切片（`text.as_bytes()[s..e]`），用 `&text[s..e]` 会因非字符边界 panic。
    let mut prev_end = 0;
    for &(s, e) in &full.offsets {
        assert!(s <= e && e <= text.len(), "offset 越界: ({s},{e})");
        assert!(s >= prev_end, "offset 必须单调不重叠");
        prev_end = e;
    }
    // round-trip：ids 解回原文（这是 offsets 语义的最终兜底）
    assert_eq!(
        tokenizer.decode(&full.ids).unwrap(),
        text,
        "decode(encode(t)) 必须无损"
    );
    // 每个 token 的字节区间内容，应与该 token 在词表中的字节一致
    for (i, &id) in full.ids.iter().enumerate() {
        let (s, e) = full.offsets[i];
        let want = &text.as_bytes()[s..e];
        let got = tokenizer
            .state
            .vocab
            .get(&id)
            .expect("token id 必须在词表中");
        // 字节级 BPE 下 token 可能只是原文的一部分（预分词可能留下未覆盖字节），
        // 因此只断言"该 token 的字节是原文 [s,e) 区间的字节"——两者必须逐字节相等
        assert_eq!(
            got.as_slice(),
            want,
            "token {id} 的字节与 offsets 区间不一致"
        );
    }
}

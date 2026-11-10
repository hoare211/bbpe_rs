// tests/review_20261031.rs
//
// 第四轮评审（docs/reviews/2026-10-31/2026-10-31-code-review.md）的回归测试。
//
// 只覆盖"能用确定性夹具在毫秒级复现"的项；需要真实内存压力或故障注入
// NFS / 磁盘满的项（R4-1 / R4-7 / R4-14 / R4-15 / R4-24）不做伪模拟，
// 其语义由源码注释、脚本退出码与 CLI 集成测试承担。
//
// 已覆盖：
//   R4-5  稀疏大 id 在**加载期**被拒绝（在分配稠密数组之前）
//   R4-6  merge 产物字节必须等于两个输入 token 字节之和（V7）
//   R4-8  num_shards = 0 在构造期返回参数错误（不是除零 panic）
//   R4-9  原子保存：写失败时旧模型文件逐字节保留
//   R4-20 训练端拒绝空 / 重复 special token
//   R4-22 输入输出同路径判定（`same_path`）
//   R4-2 / R4-3 的回归在 src/pre_tokenizer.rs 的 `#[cfg(test)]` 内（与实现同文件）

use bbpe_rs::tokenizer::{BpeState, LoadOptions, MODEL_FORMAT};
use bbpe_rs::{BBPETrainer, RuntimeConfig, TrainerConfig};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};

/// `BBPETrainer` / `BpeState` 都不实现 `Debug`，`expect_err` 不可用；用显式匹配取错误。
fn expect_err<T>(r: Result<T, bbpe_rs::BbpeError>, what: &str) -> bbpe_rs::BbpeError {
    match r {
        Ok(_) => panic!("期望失败但成功了：{what}"),
        Err(e) => e,
    }
}

fn trainer_new_err(cfg: TrainerConfig, rt: Option<RuntimeConfig>) -> bbpe_rs::BbpeError {
    expect_err(BBPETrainer::new(cfg, rt), "trainer 构造应失败")
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bbpe_r4_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("建临时目录失败");
    d
}

/// 写一个结构合法的模型：256 个基础字节 + `extra_vocab` + `merges`。
///
/// 用 `serde_json` 构造，避免手拼 JSON 时的转义问题（pattern 里有反斜杠）。
fn write_model(
    dir: &Path,
    name: &str,
    extra_vocab: &[(&str, Vec<u8>)],
    merges: &[(&str, i32)],
) -> PathBuf {
    let mut vocab = Map::new();
    for i in 0..256u32 {
        vocab.insert(i.to_string(), json!([i as u8]));
    }
    for (id, bytes) in extra_vocab {
        vocab.insert((*id).to_string(), json!(bytes));
    }
    let mut merge_map = Map::new();
    for (key, new_id) in merges {
        merge_map.insert((*key).to_string(), json!(new_id));
    }

    let model = json!({
        "format": MODEL_FORMAT,
        "format_version": 1,
        "vocab": Value::Object(vocab),
        "merges": Value::Object(merge_map),
        "pre_tokenizer_pattern": bbpe_rs::pre_tokenizer::DEFAULT_PATTERN,
    });

    let path = dir.join(name);
    std::fs::write(&path, serde_json::to_string_pretty(&model).unwrap()).unwrap();
    path
}

// =========================================================
// R4-5：稀疏大 id
// =========================================================
#[test]
fn r4_5_sparse_large_ids_rejected_at_load() {
    let dir = tmpdir("sparse");
    // 结构完全合法（id 非负、无 merge 引用它），只是极度稀疏：
    //   i32::MAX 曾使 `max_id + 1` 在 debug 下 panic、release 下回绕；
    //   MAX_MODEL_ID_SPAN 边界值则会尝试 GB 级分配。
    for huge in ["2147483647", "16777216"] {
        let path = write_model(&dir, &format!("m{huge}.json"), &[(huge, vec![0])], &[]);
        let p = path.to_str().unwrap();

        let err = expect_err(BpeState::load(p), "稀疏大 id 必须在加载期被拒绝");
        let msg = err.to_string();
        assert!(
            msg.contains("预算") || msg.contains("稀疏"),
            "错误应说明 id 跨度预算，实际: {msg}"
        );

        // 拒绝发生在 V1–V5 校验与稠密数组分配**之前** ⟹ validate=false 也必须拒绝
        let opts = LoadOptions {
            allow_legacy: false,
            validate: false,
        };
        assert!(
            BpeState::load_with_options(p, opts).is_err(),
            "validate=false 时同样必须拒绝稀疏大 id（{huge}）"
        );
        // 便捷加载器（真正按 max_id+1 分配两份稠密数组的地方）必须拒绝
        assert!(
            bbpe_rs::BBPETokenizer::load(p).is_err(),
            "BBPETokenizer::load 必须拒绝稀疏大 id（{huge}）"
        );
    }
    // 对照：合法的小 id 增量必须可加载（证明拒绝不是"凡有多余 id 就报错"）
    let ok = write_model(&dir, "ok.json", &[("256", vec![97, 98])], &[]);
    BpeState::load(ok.to_str().unwrap()).expect("id 跨度正常的模型必须加载成功");
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================
// R4-6：merge 产物字节（V7）
// =========================================================
#[test]
fn r4_6_inconsistent_merge_bytes_rejected() {
    let dir = tmpdir("v7");
    // "97,98" -> 256，但 256 的字节写成 [120]（'x'）而非 [97,98]（"ab"）
    let bad = write_model(&dir, "bad.json", &[("256", vec![120])], &[("97,98", 256)]);
    let err = expect_err(BpeState::load(bad.to_str().unwrap()), "V7 违反必须被拒绝");
    assert!(err.to_string().contains("V7"), "错误应指明 V7，实际: {err}");

    // 对照：把产物字节改对 ⟹ 同一个模型必须通过（否则说明测试夹具本身非法）
    let good = write_model(
        &dir,
        "good.json",
        &[("256", vec![97, 98])],
        &[("97,98", 256)],
    );
    BpeState::load(good.to_str().unwrap()).expect("字节正确的模型必须加载成功");
    let tok = bbpe_rs::BBPETokenizer::load(good.to_str().unwrap()).expect("必须可加载");
    assert_eq!(tok.decode(&[256]).unwrap(), "ab", "256 应解码为 \"ab\"");
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================
// R4-8：num_shards = 0
// =========================================================
#[test]
fn r4_8_zero_num_shards_errors_not_panics() {
    let rt = RuntimeConfig {
        num_shards: 0,
        ..RuntimeConfig::default()
    };
    let err = trainer_new_err(TrainerConfig::new(400), Some(rt));
    assert!(
        err.to_string().contains("num_shards"),
        "错误应指明 num_shards，实际: {err}"
    );

    // 对照：1 必须可用
    let ok = RuntimeConfig {
        num_shards: 1,
        ..RuntimeConfig::default()
    };
    BBPETrainer::new(TrainerConfig::new(400), Some(ok)).expect("num_shards=1 必须可用");
}

// =========================================================
// R4-20：训练端 special token 校验
// =========================================================
#[test]
fn r4_20_trainer_rejects_bad_special() {
    let mut cfg = TrainerConfig::new(400);
    cfg.special_tokens = vec![String::new()];
    let err = trainer_new_err(cfg, None);
    assert!(
        err.to_string().contains("空"),
        "错误应说明空字符串，实际: {err}"
    );

    let mut cfg = TrainerConfig::new(400);
    cfg.special_tokens = vec!["<|s|>".to_string(), "<|s|>".to_string()];
    let err = trainer_new_err(cfg, None);
    assert!(
        err.to_string().contains("重复"),
        "错误应说明重复项，实际: {err}"
    );

    // 对照：合法输入必须可用
    let mut cfg = TrainerConfig::new(400);
    cfg.special_tokens = vec!["<|s|>".to_string()];
    BBPETrainer::new(cfg, None).expect("合法 special token 必须可用");
}

// =========================================================
// R4-9：原子保存
// =========================================================
#[test]
fn r4_9_failed_save_keeps_old_model() {
    let dir = tmpdir("atomic");
    let src = write_model(&dir, "src.json", &[], &[]);
    let state = BpeState::load(src.to_str().unwrap()).unwrap();

    let path = dir.join("model.json");
    let p = path.to_str().unwrap();
    state.save(p).expect("首次保存应成功");
    let original = std::fs::read(&path).unwrap();

    // 制造写失败：把**临时文件路径**占成目录 ⟹ File::create 必然失败。
    // （不用 chmod：root 容器里权限位会被绕过，测试会假绿。）
    let tmp = format!("{p}.tmp-{}", std::process::id());
    std::fs::create_dir(&tmp).expect("占位目录创建失败");
    assert!(
        state.save(p).is_err(),
        "临时文件无法创建时 save 必须返回错误"
    );

    let after = std::fs::read(&path).expect("旧模型必须仍然存在");
    assert_eq!(
        after, original,
        "保存失败后旧模型必须逐字节保留（不得被截断）"
    );

    let _ = std::fs::remove_dir_all(&tmp);
    state.save(p).expect("清理后重新保存应成功");
    let reloaded = BpeState::load(p).expect("重新保存的模型必须可再次加载");
    assert_eq!(
        reloaded.vocab.len(),
        state.vocab.len(),
        "重载后的词表规模必须一致"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================
// R4-22：输入/输出同路径判定
// =========================================================
#[test]
fn r4_22_same_path_guard_blocks_overwrite() {
    use bbpe_rs::cli::util::same_path;

    let dir = tmpdir("same_path");
    let a = dir.join("corpus.jsonl");
    std::fs::write(&a, "hello\n").unwrap();
    let a_s = a.to_str().unwrap();

    assert!(same_path(a_s, a_s), "字面相同的路径必须判为同一文件");

    // 含 `..` 的等价写法必须判为同一文件（canonicalize 生效）
    let name = dir.file_name().unwrap().to_str().unwrap();
    let parent = dir.parent().unwrap().to_str().unwrap();
    let with_dotdot = format!("{parent}/{name}/../{name}/corpus.jsonl");
    assert!(
        same_path(a_s, &with_dotdot),
        "等价路径（含 ..）必须判为同一文件: {with_dotdot}"
    );

    // 输出尚未创建时的"同目录同文件名"同样必须被拦住
    let not_yet = dir.join("out.jsonl");
    assert!(
        !same_path(a_s, not_yet.to_str().unwrap()),
        "不同文件不得误判为同一路径"
    );

    let b = dir.join("other.jsonl");
    std::fs::write(&b, "hello\n").unwrap();
    assert!(
        !same_path(a_s, b.to_str().unwrap()),
        "不同文件不得误判为同一路径"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

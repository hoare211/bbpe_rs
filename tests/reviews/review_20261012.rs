// tests/review_20261012.rs
//
// 2026-10-12 第三轮评审的回归测试：
//   - C4：Rust 侧 encode 的 `allowed_special` 支持 tiktoken 式 "all" 约定
//         （与 Python 绑定 resolve_allowed_special 语义对齐）；
//   - C2：CLI 新增 `bbpe encode` / `bbpe decode` 子命令（训练产物 → 使用的闭环）。
//
// C1（空 special token 防线）的回归在 tests/model_contract_test.rs
// ::c1_empty_special_token_rejected，不在此重复。

use bbpe_rs::reader::CorpusReader;
use bbpe_rs::{BBPETrainer, TrainerConfig};
use std::path::{Path, PathBuf};
use std::process::Command;

/// 与 cli_test.rs 同源的中英混合语料（足以产生几百次合并）
const BASE_TEXT: &str = concat!(
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的",
    "能以人类智能相似的方式做出反应的智能机器。Python 是一种高级编程语言，",
    "以其简洁的语法和强大的生态系统而闻名。机器学习是人工智能的核心技术，",
    "深度学习是机器学习的重要分支。自然语言处理让计算机能够理解人类语言，",
    "大型语言模型在自然语言处理领域取得了巨大突破。The quick brown fox ",
    "jumps over the lazy dog. Machine learning models learn from large amounts ",
    "of data. Deep learning networks use multiple layers. 1234567890"
);

// special token 字面量经 concat! 拼接（避免源码中出现连续尖括号序列）
const S_START: &str = concat!("<|", "im_start", "|>");
const S_END: &str = concat!("<|", "im_end", "|>");

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bbpe_r1212_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("建临时目录失败");
    d
}

/// 训练一个含 special tokens 的小模型，返回 (模型路径, special id 列表)
fn train_model_with_specials(dir: &Path, specials: &[&str]) -> (PathBuf, Vec<i32>) {
    let corpus = dir.join("corpus.txt");
    std::fs::write(&corpus, format!("{BASE_TEXT}\n").repeat(120)).unwrap();
    let reader = CorpusReader::Text {
        filepath: corpus.to_string_lossy().into_owned(),
    };
    let mut config = TrainerConfig::new(256 + 40 + specials.len());
    config.special_tokens = specials.iter().map(|s| s.to_string()).collect();
    let mut trainer = BBPETrainer::new(config, None).unwrap();
    trainer.train(&reader).unwrap();

    let ids: Vec<i32> = specials
        .iter()
        .map(|s| trainer.state.special_tokens[*s])
        .collect();
    let path = dir.join("vocab_specials.json");
    trainer.state.save(path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_file(&corpus);
    (path, ids)
}

fn bbpe() -> Command {
    Command::new(env!("CARGO_BIN_EXE_bbpe"))
}

/// 跑 `bbpe <args>`，断言成功并返回 stdout
fn run_ok(args: &[&str]) -> String {
    let out = bbpe().args(args).output().expect("运行 bbpe 失败");
    assert!(
        out.status.success(),
        "bbpe {} 失败: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// 跑 `bbpe <args>` 并解析 stdout 的 JSON
fn run_json(args: &[&str]) -> serde_json::Value {
    let stdout = run_ok(args);
    serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("bbpe {} 输出不是合法 JSON: {e}\n{stdout}", args.join(" ")))
}

// =========================================================
// C4：Rust 侧 `allowed_special` 的 "all" 约定
// =========================================================
#[test]
fn c4_rust_encode_allowed_special_all_convention() {
    let dir = tmpdir("c4");
    let specials: [&str; 2] = [S_START, S_END];
    let (path, ids) = train_model_with_specials(&dir, &specials);
    let (id_start, id_end) = (ids[0], ids[1]);

    let tok = bbpe_rs::BBPETokenizer::load(path.to_str().unwrap()).unwrap();
    // 文本 > 5000 字节 ⟹ encode_parallel 走 rayon 分支，顺带守护并行同源
    let body = BASE_TEXT.repeat(8);
    let text = format!("{S_START}{body}{S_END}");

    // ① {"all"} ⟹ 全部 special 放行（整体成词）
    let all: std::collections::HashSet<&str> = ["all"].into_iter().collect();
    let enc = tok.encode(&text, &all, false);
    assert!(
        enc.ids.contains(&id_start) && enc.ids.contains(&id_end),
        "allowed_special={{\"all\"}} 必须放行全部 special（C4）"
    );

    // ② {"im_start", "all"} ⟹ 仍全部放行（任一项为 "all" 即生效，与 Python 侧同口径）
    let mixed: std::collections::HashSet<&str> = [S_START, "all"].into_iter().collect();
    let enc_mixed = tok.encode(&text, &mixed, false);
    assert_eq!(enc_mixed.ids, enc.ids, "\"all\" 与具体项混用时必须全放行");

    // ③ 并行路径与串行一致
    let enc_par = tok.encode_parallel(&text, &all, false, Some(2));
    assert_eq!(
        enc_par.ids, enc.ids,
        "encode_parallel 与 encode 在 \"all\" 下必须一致"
    );

    // ④ 对照：白名单不含 "all" ⟹ 一个都不放行（证明 ① 生效的不是别的因素）
    let none: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let enc_none = tok.encode(&text, &none, false);
    assert!(
        !enc_none.ids.contains(&id_start) && !enc_none.ids.contains(&id_end),
        "未授权 special 不得整体成词（防注入）"
    );

    // ⑤ decode 往返无损
    assert_eq!(
        tok.decode(&enc.ids).unwrap(),
        text,
        "decode(encode(t)) 必须无损"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================
// C2：CLI `bbpe encode` / `bbpe decode` 端到端
// =========================================================
#[test]
fn c2_cli_encode_decode_roundtrip() {
    let dir = tmpdir("c2");

    // ① 用 CLI 训练一个模型（与 cli_test.rs 同口径）
    let corpus = dir.join("corpus.jsonl");
    let mut lines = String::new();
    for i in 0..300 {
        lines.push_str(&format!("{{\"text\": \"{BASE_TEXT} #{i}\"}}\n"));
    }
    std::fs::write(&corpus, lines).unwrap();
    let model = dir.join("vocab.json");
    run_ok(&[
        "train",
        "--input",
        corpus.to_str().unwrap(),
        "--format",
        "jsonl",
        "--vocab",
        "520",
        "--output",
        model.to_str().unwrap(),
    ]);

    let text = "人工智能 hello world";
    let m = model.to_str().unwrap();

    // ② encode --text：人类可读输出含 ids 行
    let stdout = run_ok(&["encode", "--model", m, "--text", text]);
    assert!(
        stdout.contains("ids:"),
        "encode 输出应含 ids 行，实际: {stdout}"
    );

    // ③ encode --json ⟹ 取 ids ⟹ decode 必须还原原文
    let v = run_json(&["encode", "--model", m, "--text", text, "--json"]);
    let ids: Vec<i32> = v["ids"]
        .as_array()
        .expect("JSON 应含 ids 数组")
        .iter()
        .map(|x| x.as_i64().expect("id 应为整数") as i32)
        .collect();
    assert!(!ids.is_empty(), "ids 不应为空");
    let ids_str = ids
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let decoded = run_ok(&["decode", "--model", m, "--ids", &ids_str]);
    assert_eq!(decoded.trim_end(), text, "CLI decode(encode(t)) 必须无损");

    // ④ encode --file 与 --text 结果一致
    let f = dir.join("dev.txt");
    std::fs::write(&f, text).unwrap();
    let v_file = run_json(&[
        "encode",
        "--model",
        m,
        "--file",
        f.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(v_file["ids"], v["ids"], "--file 与 --text 必须给出一致结果");

    // ⑤ special token 白名单三态：默认切碎 / --allowed-special 放行 / --all-special 放行
    let specials: [&str; 2] = [S_START, S_END];
    let (smodel, sids) = train_model_with_specials(&dir, &specials);
    let sm = smodel.to_str().unwrap();
    let stext = format!("{S_START}{text}{S_END}");

    let v_def = run_json(&["encode", "--model", sm, "--text", &stext, "--json"]);
    let def_ids: Vec<i32> = v_def["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    assert!(
        !def_ids.contains(&sids[0]) && !def_ids.contains(&sids[1]),
        "默认（无白名单）special 必须被切碎（防注入）"
    );

    let allowed_arg = format!("{S_START},{S_END}");
    let v_wl = run_json(&[
        "encode",
        "--model",
        sm,
        "--text",
        &stext,
        "--allowed-special",
        &allowed_arg,
        "--json",
    ]);
    let wl_ids: Vec<i32> = v_wl["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    assert!(
        wl_ids.contains(&sids[0]) && wl_ids.contains(&sids[1]),
        "--allowed-special 白名单内的 special 必须整体成词"
    );

    let v_all = run_json(&[
        "encode",
        "--model",
        sm,
        "--text",
        &stext,
        "--all-special",
        "--json",
    ]);
    let all_ids: Vec<i32> = v_all["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    assert!(
        all_ids.contains(&sids[0]) && all_ids.contains(&sids[1]),
        "--all-special 必须放行全部 special"
    );
    assert_eq!(all_ids, wl_ids, "--all-special 与全量白名单应给出一致结果");

    // ⑥ 错误路径：缺 --ids / 未知 id ⟹ 非零退出且 stderr 有信息（不得 panic）
    let out = bbpe()
        .args(["decode", "--model", m, "--ids", "999999"])
        .output()
        .expect("运行 bbpe 失败");
    assert!(!out.status.success(), "未知 token id 必须报错");
    assert!(
        !String::from_utf8_lossy(&out.stderr).is_empty(),
        "报错时 stderr 必须有信息"
    );

    let out = bbpe()
        .args(["decode", "--model", m])
        .output()
        .expect("运行 bbpe 失败");
    assert!(!out.status.success(), "缺 --ids 必须报错");

    let out = bbpe()
        .args(["encode", "--model", m])
        .output()
        .expect("运行 bbpe 失败");
    assert!(!out.status.success(), "缺 --text/--file 必须报错");

    let _ = std::fs::remove_dir_all(&dir);
}

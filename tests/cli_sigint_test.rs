//! R6-1：CLI 的 SIGINT 优雅取消。
//!
//! 原生二进制默认 SIGINT 直接终止进程，`TempFileGuard::drop` 不会运行。
//! 注册 handler 后信号变成 `CancelToken`，core 检查点返回 `Err(Cancelled)`，
//! CLI 正常展开栈：进度条 abandon、临时文件由 RAII 清理、原始语料不动。
//!
//! 用 `CARGO_BIN_EXE_bbpe` 让 Cargo 自动构建被测二进制；用系统的 `kill`
//! 命令发信号，避免为此引入 libc/nix 直接依赖。

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

static SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_dir() -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!(
        "bbpe_cli_sigint_{}_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        n
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

fn sample_temp_files() -> std::collections::HashSet<PathBuf> {
    let mut out = std::collections::HashSet::new();
    let Ok(entries) = fs::read_dir(std::env::temp_dir()) else {
        return out;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with("bbpe_train_sample_") {
            out.insert(e.path());
        }
    }
    out
}

#[test]
fn cli_ctrl_c_cancels_during_sampling() {
    let dir = tmp_dir();
    let corpus = dir.join("corpus.txt");
    let model = dir.join("model.json");
    {
        let mut f = fs::File::create(&corpus).unwrap();
        for i in 0..800_000 {
            writeln!(
                f,
                "中文分词训练样本 序号{i} abcdef0123456789 语言文字处理 字节级BPE"
            )
            .unwrap();
        }
    }

    let before = sample_temp_files();
    let child = Command::new(env!("CARGO_BIN_EXE_bbpe"))
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--format",
            "text",
            "--vocab",
            "2000",
            "--output",
            model.to_str().unwrap(),
            "--sample-rate",
            "0.5",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("启动 bbpe train 失败");

    // 等采样临时文件出现 ⟹ 已进入采样预读阶段
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut appeared = false;
    while std::time::Instant::now() < deadline {
        if !sample_temp_files().is_subset(&before) {
            appeared = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(appeared, "未观察到采样临时文件，无法定位采样阶段");

    let t0 = std::time::Instant::now();
    let status = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("无法发送 SIGINT");
    assert!(status.success(), "kill -INT 失败");

    let out = child.wait_with_output().expect("等待子进程失败");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "取消后应非零退出：{combined}");
    assert!(
        combined.contains("Ctrl+C") || combined.contains("取消"),
        "缺少取消信息：{combined}"
    );
    assert!(!model.exists(), "取消后不能产出模型");
    assert!(Path::new(&corpus).exists(), "原始语料不得被删除");
    assert!(
        t0.elapsed() < Duration::from_secs(10),
        "采样阶段取消未及时生效"
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if sample_temp_files().is_subset(&before) {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        sample_temp_files().is_subset(&before),
        "临时采样文件残留：{:?}",
        sample_temp_files().difference(&before).collect::<Vec<_>>()
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn cli_ctrl_c_leaves_no_partial_model() {
    let dir = tmp_dir();
    let corpus = dir.join("corpus.txt");
    let model = dir.join("model.json");

    // 每行唯一，避免词表过小导致训练瞬间结束；总量约 40MB，训练足以持续数秒
    {
        let mut f = fs::File::create(&corpus).unwrap();
        for i in 0..500_000 {
            writeln!(
                f,
                "中文分词训练样本 序号{i} abcdef0123456789 语言文字处理 字节级BPE"
            )
            .unwrap();
        }
    }

    let bin = env!("CARGO_BIN_EXE_bbpe");
    let child = Command::new(bin)
        .args([
            "train",
            "--input",
            corpus.to_str().unwrap(),
            "--format",
            "text",
            "--vocab",
            "2000",
            "--output",
            model.to_str().unwrap(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("启动 bbpe train 失败");

    // 等训练真正进入 Rust 主循环
    std::thread::sleep(Duration::from_secs(2));
    let status = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("无法发送 SIGINT");
    assert!(status.success(), "kill -INT 失败");

    let out = child.wait_with_output().expect("等待子进程失败");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = format!("{stdout}{stderr}");

    assert!(
        !out.status.success(),
        "取消后 CLI 应以非零码退出；输出：{combined}"
    );
    assert!(
        combined.contains("Ctrl+C") || combined.contains("取消"),
        "缺少明确的取消信息：{combined}"
    );
    assert!(
        !model.exists(),
        "取消后不能产出半截模型（train_inner 不得写 meta/save）"
    );
    assert!(
        Path::new(&corpus).exists(),
        "R4-1 铁律：取消/失败分支绝不能删除原始语料"
    );

    let _ = fs::remove_dir_all(dir);
}

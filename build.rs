// 模型契约 v1（§7.1）：把 git commit 注入编译产物，写入 meta.git_commit（G1 可追溯）。
fn main() {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_COMMIT={commit}");
    println!("cargo:rerun-if-changed=build.rs");
    // Cargo 语义：一旦 build script 发出任何 `rerun-if-changed`，就**只**监视
    // 列出的文件。此前只列了 build.rs 自身，于是 `git commit` 之后
    // `cargo build` 不会重跑本脚本，GIT_COMMIT 冻结在旧值 ——
    // 直接削弱 G1 可追溯（模型 meta 记录了一个不再对应源码的 commit）。
    //
    // B1：A6 的修法（只监视 `.git/HEAD`）**无效**，比不修更危险 —— 它写了注释、
    // 加了代码、却从未验证过，会让后来者默认它是好的。
    // 原因：`.git/HEAD` 的内容是 `ref: refs/heads/<branch>`，**只在切分支 /
    // detach 时改变**；普通 `git commit` 改的是它指向的那个 ref 文件
    // （`.git/refs/heads/<branch>`），HEAD 本身纹丝不动。
    //
    // 本项目自身的证据（2026-10-07 实测）：
    //   .git/HEAD                              mtime = 2026-09-08 10:21:07
    //   .git/refs/heads/feat/track-a-sampling  mtime = 2026-10-07 22:51:33
    // 当天 13 个整改提交（22:49:54 → 22:51:33）没有一次碰过 .git/HEAD；那时
    // `meta.git_commit` 之所以恰好正确，是因为 Cargo.toml 变更引发了 build.rs
    // 重跑 —— 不是 A6 的功劳。
    //
    // 受控实验（临时仓库 + 同款 build.rs，跑完已删除）：
    //   只监视 HEAD：c1 构建 → 空提交 c2 → 再构建，仍打印 c1 ❌
    //   改为监视 HEAD 指向的 ref：c3 → f82d182 ✅、c4 → 5290c88 ✅
    //
    // `packed-refs` 那条不能省：`git gc` 会把 loose ref 打包进该文件，
    // 只监视 branch 文件会再次失效。
    //
    // 注意：打包发布（无 .git 目录）时这些路径不存在，`rerun-if-changed` 指向
    // 缺失文件只会让 cargo 跳过重跑，git 命令失败已被上面的 `unwrap_or`
    // 兜底为 "unknown"，不会破坏 CI。
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(r) = head.trim().strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed=.git/{r}");
        }
    }
    println!("cargo:rerun-if-changed=.git/packed-refs");
}

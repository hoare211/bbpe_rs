// src/bin/compare_merges.rs —— **过渡别名**，已并入统一入口 `bbpe compare`（参数不变）
fn main() {
    eprintln!("⚠️  compare_merges 已并入统一入口：请改用 `bbpe compare ...`（参数不变）");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::compare::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

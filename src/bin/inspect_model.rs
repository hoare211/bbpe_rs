// src/bin/inspect_model.rs —— **过渡别名**，已并入统一入口 `bbpe inspect`（参数不变）
fn main() {
    eprintln!("⚠️  inspect_model 已并入统一入口：请改用 `bbpe inspect ...`（参数不变）");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::inspect::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

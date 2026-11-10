// src/bin/bbpe.rs —— 统一入口：`bbpe <子命令> [参数]`
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::dispatch::run(&args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

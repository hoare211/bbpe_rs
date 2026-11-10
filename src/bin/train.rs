// src/bin/train.rs —— 等价于 `bbpe train ...`（保留独立调用习惯）
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::train::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

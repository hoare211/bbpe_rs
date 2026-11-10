// src/bin/compare.rs —— 等价于 `bbpe compare ...`
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::compare::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

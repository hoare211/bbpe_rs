// src/bin/sample.rs —— 等价于 `bbpe sample ...`
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::sample::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

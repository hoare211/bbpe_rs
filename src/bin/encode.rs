// src/bin/encode.rs —— 等价于 `bbpe encode ...`
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::encode::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

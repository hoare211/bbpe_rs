// src/bin/decode.rs —— 等价于 `bbpe decode ...`
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::decode::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

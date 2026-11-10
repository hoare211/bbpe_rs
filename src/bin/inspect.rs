// src/bin/inspect.rs —— 等价于 `bbpe inspect ...`
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::inspect::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

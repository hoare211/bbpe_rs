// src/bin/sweep.rs —— 等价于 `bbpe sweep ...`
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::sweep::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

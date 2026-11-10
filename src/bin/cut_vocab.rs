// src/bin/cut_vocab.rs —— **过渡别名**，已并入统一入口 `bbpe cut`
// `--vocab` 一直是总词表口径，参数不变。
fn main() {
    eprintln!("⚠️  cut_vocab 已并入统一入口：请改用 `bbpe cut ...`（参数不变）");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::cut::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

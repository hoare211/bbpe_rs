// src/bin/sample_corpus.rs —— **过渡别名**，已并入统一入口 `bbpe sample`
// 注意：日常训练不再需要单独采样，`bbpe train --sample-rate ...` 已内建采样。
fn main() {
    eprintln!("⚠️  sample_corpus 已并入统一入口：请改用 `bbpe sample ...`（参数不变）");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::sample::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

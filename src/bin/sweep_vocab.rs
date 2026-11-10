// src/bin/sweep_vocab.rs —— **过渡别名**，已并入统一入口 `bbpe sweep`
//
// ⚠️ 单位变更：`--vocabs` 由 "merges 数 K" 改为 **总词表**（256 + K），
//    与 train / cut 同口径。旧值 100000 现在应写 100256。
fn main() {
    eprintln!("⚠️  sweep_vocab 已并入统一入口：请改用 `bbpe sweep ...`");
    eprintln!("⚠️  注意 --vocabs 单位已改为总词表（256 + K），旧值需 +256");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::sweep::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

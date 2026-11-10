// src/bin/train_vocab.rs —— **过渡别名**，已并入统一入口 `bbpe train`
//
// ⚠️ 行为变更（这是有意的修复）：旧 `--sampling-rate` 只把数字写进 meta，
//    并不真的采样；现在 `train` 会按该参数**真采样**（出身信息因此才是事实）。
//    若旧脚本依赖"传了 --sampling-rate 但喂的是全量语料"，请去掉该参数。
fn main() {
    eprintln!("⚠️  train_vocab 已并入统一入口：请改用 `bbpe train ...`（参数完全兼容）");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bbpe_rs::cli::train::run(args) {
        eprintln!("❌ {e}");
        std::process::exit(1);
    }
}

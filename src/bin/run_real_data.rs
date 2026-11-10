// src/bin/run_real_data.rs
use bbpe_rs::{BBPETokenizer, BBPETrainer, TrainerConfig};
use tracing::info;
// m3/n5：原 `use bbpe_rs::CorpusReader::{JsonlStaticConversations, JsonlStaticText};`
// 只用到后者；`use dhat::{Profiler};` 对应的 Profiler 与 #[global_allocator] 均已
// 注释，属死导入。二者一并清理（dhat 依赖也已从 Cargo.toml 移除）。
use bbpe_rs::CorpusReader::JsonlStaticText;

// #[global_allocator]
// static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let _profiler = dhat::Profiler::new_heap();

    tracing_subscriber::fmt().with_target(false).init();

    let jsonl_path = "data/corpus.jsonl";
    let model_path = "output/model.json";

    info!("启动纯 Rust 极速训练 (直接流式解析 JSONL)...");

    // 【质变】：直接实例化 JSONL Reader，无需任何预处理转存文件！
    let reader = JsonlStaticText {
        filepath: jsonl_path.to_string(),
    };
    // let reader = JsonlStaticConversations { filepath: jsonl_path.to_string(), sep: "\n".to_string() };

    let config = TrainerConfig::new(6400);
    let mut trainer = BBPETrainer::new(config, None)?;
    // trainer.train(&reader, 1000, Some(4))?;
    trainer.train(&reader)?;

    // m12：save 前确保父目录存在。原实现直接 `File::create("output/...")`，
    // 在干净工作区（无 output/）会以 NotFound 失败，且错误信息不指向根因。
    if let Some(dir) = std::path::Path::new(model_path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    trainer.save(model_path)?;

    info!("启动纯 Rust 推断验证...");
    let tokenizer = BBPETokenizer::load(model_path)?;
    // let test_str = "无为而治是一项清净的法则。";
    let test_str = "Python 是一种 高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is   widely  used in data science, machine  learning, and web  development. 开发者可以利用  NumPy,  Pandas ,  and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来 就像英 语一样。Whether you are a beginner or an expert, Python offers something for everyone.";
    let ids = tokenizer.encode_ordinary(test_str);

    info!("原文: {}", test_str);
    info!("Tokens IDs: {:?}", ids);
    info!("Decode 还原: {}", tokenizer.decode(&ids)?);

    Ok(())
}

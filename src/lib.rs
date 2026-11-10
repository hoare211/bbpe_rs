pub mod cancel;
pub mod compare;
pub mod error;
pub mod external;
pub mod pre_tokenizer;
pub mod tokenizer;
pub mod trainer;

pub use cancel::{CancelReason, CancelToken};
pub use error::BbpeError;
pub use tokenizer::BBPETokenizer;
pub use trainer::BBPETrainer;
pub use trainer::RuntimeConfig;
pub use trainer::TrainerConfig;

pub mod bindings;
pub mod reader;
pub mod sampling;

// 统一命令行实现（供 `bbpe` 与各单命令 bin 共用，避免行为分叉）
pub mod cli;

pub use reader::{CorpusReader, CorpusStats};

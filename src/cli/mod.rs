// src/cli/mod.rs
//
// 统一命令行界面（"三支柱"方案的落地处）。
//
// 为什么集中到 lib 里：统一入口 `bbpe <子命令>` 需要复用各工具的实现，
// 而 `src/bin/*.rs` 之间无法互相引用。把实现放进 lib 后：
//   - `bbpe train ...`   —— 唯一推荐入口（子命令风格）
//   - `train ...` / `train_vocab ...` —— 仅是一层 4 行的转发壳，方便老脚本过渡
// 二者调用的是同一份代码，不存在行为分叉。
//
// 三支柱：
//   1. **单一入口 train**：采样内建。不传 `--sample-rate` 就是全量精确训练；
//      传了就在训练前做行级流式采样，出身信息由程序据实写入（不靠人工转录）。
//   2. **单位统一**：所有用户可见的词表规模一律是**总词表** = 256 + K。
//      `train` / `sweep` / `cut` 三处口径一致，换算只在 `util::vocab_to_k` 一处发生。
//   3. **零剖面**：`track-a` feature 已删除，所有工具一次编译全部产出；
//      发布形态（完整包 / v2 演示包）只是"发哪几个 bin"的打包差异。

pub mod compare;
pub mod cut;
pub mod decode;
pub mod dispatch;
pub mod encode;
pub mod inspect;
pub mod sample;
pub mod sweep;
pub mod train;
pub mod util;

pub use dispatch::run;

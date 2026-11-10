//! 历史评审回归测试的**单一入口**。
//!
//! 为什么合并：`tests/<name>.rs` 每个文件都会被 cargo 编译成一个**独立的测试
//! 二进制**，各自完整链接一遍 lib。评审文件按月累积（2026-10-07 / 10-12 /
//! 10-31 / 11-01 / 11-07 …），每个二进制在 debug 下 50MB+、release 下 35MB，
//! 是 `target/` 膨胀的主要来源之一。
//!
//! 放在 `tests/reviews/` 子目录 + 本文件后，cargo 只会识别 `tests/reviews/main.rs`
//! 这一个测试目标（名为 `reviews`），各期文件作为模块编入同一个二进制。
//!
//! 新增一期评审时：
//!   1. 新建 `tests/reviews/review_YYYYMMDD.rs`（**不要**放回 `tests/` 根目录，
//!      否则又会变成独立的测试二进制）；
//!   2. 在下面按时间顺序补一行 `mod review_YYYYMMDD;`。
//!
//! 运行：`cargo test --test reviews`（或按用例名过滤 `cargo test --test reviews r6_`）。

mod review_20261007;
mod review_20261012;
mod review_20261031;
mod review_20261101;
mod review_20261107;

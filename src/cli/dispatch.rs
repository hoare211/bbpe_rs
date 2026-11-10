// src/cli/dispatch.rs
//
// `bbpe` 统一入口的子命令分发。
//
// 所有子命令的实现都在本模块树里（`train` / `sample` / `sweep` / `cut` / `inspect` / `compare` / `encode` / `decode`），
// 旧的单命令 bin（`train_vocab` 等）只是一层转发壳 ⟹ **同一份代码，不存在行为分叉**。

use super::util::CliResult;
use super::{compare, cut, decode, encode, inspect, sample, sweep, train};

pub const USAGE: &str = "\
bbpe —— byte-level BPE 工具链（统一入口）

用法:
  bbpe <子命令> [参数...]

子命令:
  train     训练词表（不采样 = 全量精确训练；加 --sample-rate = 大语料采样训练）
  sweep     一次训练扫全部词表规模，用 held-out 选 K（可 --export-at 直接导出）
  cut       从 K_max 模型导出任意规模（零重训）
  sample    单独采一次样，供多次训练复用（进阶）
  compare   比对两条 merges 序列，判断采样率/语料规模是否足够
  inspect   模型体检 / 校验 / v0→v1 迁移
  encode    用模型编码文本（--text / --file），支持 special token 白名单
  decode    用模型把 id 序列解码回文本（encode 的逆操作）

词表规模单位（全部一致）:
  总词表 = 256 基础字节 + merges 数 K。train / sweep / cut 三处同口径。

示例:
  bbpe train --input data/train.jsonl --vocab 2000 --output output/vocab.json
  bbpe encode --model output/vocab.json --text '你好，世界'
  bbpe encode --model output/vocab.json --file data/dev.txt --json
  bbpe decode --model output/vocab.json --ids 1,2,3,4
  bbpe inspect output/vocab.json --validate-only

提示: 每个子命令也可单独调用（train / encode / ...），参数完全相同。
";

pub fn run(args: &[String]) -> CliResult {
    super::util::init_tracing_once();
    if args.is_empty() {
        eprint!("{USAGE}");
        std::process::exit(2);
    }
    if matches!(args[0].as_str(), "-h" | "--help" | "help") {
        print!("{USAGE}");
        return Ok(());
    }
    let rest: Vec<String> = args[1..].to_vec();
    match args[0].as_str() {
        "train" => train::run(rest),
        "sample" => sample::run(rest),
        "sweep" => sweep::run(rest),
        "cut" => cut::run(rest),
        "inspect" => inspect::run(rest),
        "compare" => compare::run(rest),
        "encode" => encode::run(rest),
        "decode" => decode::run(rest),
        other => Err(format!("未知子命令: {other}\n\n{USAGE}").into()),
    }
}

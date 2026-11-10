// src/sampling.rs
//
// 轨 A 行级均匀随机采样核心（从 sample_corpus bin 抽取为库函数，供回归测试共用）。
//
// 设计要点（依据方案 §19.1 / 小黑板 2026-09-04（六）轨 A 参数规格）：
//   - **禁止取前 N 行**：语料按来源排序会引入严重偏置；必须全语料均匀随机。
//   - **行级而非字节级**：换行是预分词边界，行级采样不切词，保证语义完整。
//   - **固定种子**：splitmix64，同一 (seed, rate) 必得同一样本，实验可复现。
//   - **训练/held-out 互斥**：同一行先掷 held-out 骰子，命中则写入 held-out 且
//     绝不进入训练集，两者严格不相交。
//   - **嵌套性保证（F18，2026-09-07 夜修复）**：trng / hrng **每一行都消耗**，
//     与 held-out 是否命中、held-out 是否启用无关。⟹ 同一 seed 下
//     `rate₁ < rate₂` 的训练集**严格嵌套**（rate₁ ⊆ rate₂），且 held-out 集合
//     与 rate 无关。修复前 held-out 命中行会跳过 trng ⟹ 同 seed 不同配置的
//     样本互相独立，曾导致"嵌套对照"实验误判。
//   - **--max-bytes 与 rate 无关**：截断点只依赖累计字节数 ⟹ 嵌套性在截断下也保持。
//
// 本模块的正确性由 tests/track_a_test.rs 常驻守护（嵌套性/互斥性/held-out 稳定性）。

use std::io::{BufRead, Write};

/// splitmix64：小而快的确定性 PRNG，同种子必然复现。
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// [0, 1) 均匀分布（53 位精度，足够 rate 判定使用）
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// 采样统计（供调用方打印 / 测试断言）。
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SampleStats {
    pub total_lines: u64,
    pub total_bytes: u64,
    pub train_lines: u64,
    pub train_bytes: u64,
    pub held_lines: u64,
    pub held_bytes: u64,
}

/// 采样参数（`sample_lines` 的配置，避免 8 个位置参数）。
/// A1(clean)：原先 8 个位置参数触发 clippy::too_many_arguments，
/// 且调用方（cli/train、cli/sample）只能靠位置记忆，易传错顺序。
#[derive(Debug, Clone, Copy)]
pub struct SampleParams {
    /// 训练集采样率 ∈ [0,1]
    pub rate: f64,
    /// 训练集采样 PRNG 种子
    pub seed: u64,
    /// held-out 采样率 ∈ [0,1]（0 = 不启用）
    pub heldout_rate: f64,
    /// held-out 采样 PRNG 种子（与 seed 独立）
    pub heldout_seed: u64,
    /// 累计读入字节上限（达到即停，行完整保留）
    pub max_bytes: u64,
}

/// 行级均匀随机采样。
///
/// 语义（与 sample_corpus bin 的历史行为逐条一致，仅参数校验收紧）：
///   1. 每行先掷 `u_held`，若 `heldout_rate > 0 && u_held < heldout_rate`：
///      行写入 held_out（若提供）且**绝不**进入训练集；
///   2. 否则掷 `u_train`，`u_train < rate` 的行写入训练集；
///   3. 两个 rng 每行都推进（F18 嵌套性保证，见模块注释）；
///   4. 累计读入字节达到 `max_bytes` 时停止（行完整保留，不产生半行）。
///
/// 错误：rate 不在 [0,1]、`heldout_rate > 0` 但未提供 held_out writer 时返回
/// `InvalidInput`（后者旧行为是静默丢行——既不进训练集也不落盘，属配置错误）。
pub fn sample_lines<R, W>(
    reader: &mut R,
    train_out: &mut W,
    held_out: Option<&mut W>,
    p: SampleParams,
) -> std::io::Result<SampleStats>
where
    R: BufRead,
    W: Write,
{
    sample_lines_cancel(reader, train_out, held_out, p, None)
}

/// R6-1 P3 补遗：带取消令牌的采样。
///
/// 原 [`sample_lines`] 转调本函数（`cancel=None`），因此 CLI 与既有测试的
/// 行为逐位不变。绑定层把采样放进 worker 后，主线程信号泵置位令牌，
/// 本循环每 4096 行检查一次并提前退出 —— 200GB 语料的预读阶段也能被
/// Ctrl+C 停止，而不是等预读跑完才响应。
pub fn sample_lines_cancel<R, W>(
    reader: &mut R,
    train_out: &mut W,
    mut held_out: Option<&mut W>,
    p: SampleParams,
    cancel: Option<&crate::cancel::CancelToken>,
) -> std::io::Result<SampleStats>
where
    R: BufRead,
    W: Write,
{
    let SampleParams {
        rate,
        seed,
        heldout_rate,
        heldout_seed,
        max_bytes,
    } = p;
    if !(0.0..=1.0).contains(&rate) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("rate 必须在 [0,1]，实际 {rate}"),
        ));
    }
    if !(0.0..=1.0).contains(&heldout_rate) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("heldout_rate 必须在 [0,1]，实际 {heldout_rate}"),
        ));
    }
    if heldout_rate > 0.0 && held_out.is_none() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "heldout_rate > 0 时必须提供 held-out 输出（否则命中行会被静默丢弃）",
        ));
    }

    let mut trng = SplitMix64::new(seed);
    let mut hrng = SplitMix64::new(heldout_seed);
    let mut stats = SampleStats::default();
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);

    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }
        stats.total_lines += 1;
        stats.total_bytes += n as u64;

        // R6-1 P3：预读阶段的取消检查点（4096 行 ≈ 毫秒级响应）。
        // 用 ErrorKind::Interrupted 与真实 IO 故障区分，调用方据此映射成取消。
        if stats.total_lines % 4096 == 0 {
            if let Some(c) = cancel {
                if c.is_cancelled() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "采样被取消",
                    ));
                }
            }
        }

        // 关键：① 先掷 held-out —— 命中则绝不进入训练集，两者严格不相交；
        //       ② 两个 rng **每行都推进**（F18），否则 trng 序列会随 held-out
        //          命中情况漂移，同 seed 不同 rate 不再嵌套，标定实验会失真。
        let u_train = trng.next_f64();
        let u_held = hrng.next_f64();
        if heldout_rate > 0.0 && u_held < heldout_rate {
            stats.held_lines += 1;
            stats.held_bytes += n as u64;
            if let Some(w) = held_out.as_mut() {
                w.write_all(&buf)?;
            }
        } else if u_train < rate {
            stats.train_lines += 1;
            stats.train_bytes += n as u64;
            train_out.write_all(&buf)?;
        }

        if stats.total_bytes >= max_bytes {
            break;
        }
    }
    train_out.flush()?;
    if let Some(w) = held_out.as_mut() {
        w.flush()?;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancel::{CancelReason, CancelToken};
    use std::io::BufReader;

    fn params() -> SampleParams {
        SampleParams {
            rate: 1.0,
            seed: 42,
            heldout_rate: 0.0,
            heldout_seed: 43,
            max_bytes: u64::MAX,
        }
    }

    #[test]
    fn r6_1_sampling_loop_honors_cancel_token() {
        let token = CancelToken::new();
        token.cancel(CancelReason::User);
        let text = "中文采样测试 abc\n".repeat(10_000);
        let mut input = BufReader::new(std::io::Cursor::new(text.into_bytes()));
        let mut out: Vec<u8> = Vec::new();
        let err = sample_lines_cancel(
            &mut input,
            &mut out,
            None::<&mut Vec<u8>>,
            params(),
            Some(&token),
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Interrupted);
    }

    #[test]
    fn r6_1_matches_sample_lines_when_not_cancelled() {
        let text = "中文采样测试 abc\n".repeat(10_000);
        let mut a: Vec<u8> = Vec::new();
        let mut b: Vec<u8> = Vec::new();
        let s1 = sample_lines(
            &mut BufReader::new(std::io::Cursor::new(text.as_bytes())),
            &mut a,
            None::<&mut Vec<u8>>,
            params(),
        )
        .unwrap();
        let token = CancelToken::new();
        let s2 = sample_lines_cancel(
            &mut BufReader::new(std::io::Cursor::new(text.as_bytes())),
            &mut b,
            None::<&mut Vec<u8>>,
            params(),
            Some(&token),
        )
        .unwrap();
        assert_eq!(s1, s2);
        assert_eq!(a, b);
    }
}

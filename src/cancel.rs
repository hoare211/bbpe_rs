//! 训练取消令牌（R6-1）。
//!
//! 设计边界：**core 只认这个纯 Rust 类型，不知道 Python/GIL/信号的存在**。
//! 「信号如何变成取消意图」由绑定层决定：P1 起主线程 pump loop 调
//! `py.check_signals()` 后直接 `cancel()`；`probe` 保留给纯 Rust 调用方
//! 和单测。训练循环只做 `load(Relaxed)`，因此细粒度检查点几乎零开销。
//! `progress` 让主线程在取消时无需 join worker 就能读到已完成 merge 数。

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// 进程内的单调微秒时钟基准（只用于探测节流，不用于训练语义）。
static EPOCH: OnceLock<Instant> = OnceLock::new();

fn now_us() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

/// 取消原因。纯 Rust 枚举，便于跨线程传递；绑定层据此决定抛什么异常。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReason {
    None,
    /// Python 侧信号（默认 SIGINT → KeyboardInterrupt，也可能是自定义 handler 抛的异常）
    Signal,
    /// 调用方显式 `cancel()`
    User,
    /// 超时
    Timeout,
}

impl CancelReason {
    fn as_u8(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Signal => 1,
            Self::User => 2,
            Self::Timeout => 3,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Signal,
            2 => Self::User,
            3 => Self::Timeout,
            _ => Self::None,
        }
    }
}

impl std::fmt::Display for CancelReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::None => "未取消",
            Self::Signal => "信号中断",
            Self::User => "调用方取消",
            Self::Timeout => "超时",
        })
    }
}

/// 训练取消令牌。
///
/// - `flag` / `reason`：`Release`/`Acquire` 配对，保证置位原因先于标志可见。
/// - `probe`：可选的**粗粒度**信号探测钩子，返回 `true` 表示应取消。
///   由 `poll()` 按 `probe_interval_us` 节流调用；细粒度检查点用
///   `is_cancelled()` 只读标志，不触发探测。
/// - `probe` 要求 `Send + Sync`，使 `CancelToken` 在后续阶段可跨线程共享；
///   绑定层构造时不得捕获 `Python<'py>` 等 GIL 绑定类型。
pub struct CancelToken {
    flag: AtomicBool,
    reason: AtomicU8,
    /// 最近一次检查点看到的已完成 merge 数。由 core 在生成 Cancelled 时写入，
    /// 绑定层的主线程可在不 join worker 的情况下读取用于异常 provenance。
    progress: AtomicUsize,
    probe: Option<Box<dyn Fn() -> bool + Send + Sync>>,
    last_probe_us: AtomicU64,
    probe_interval_us: u64,
}

impl CancelToken {
    /// 由外部（主线程 pump loop / 其它线程）置位的令牌。
    pub fn new() -> Self {
        Self::with_probe(None, Duration::from_millis(50))
    }

    pub fn with_probe(
        probe: Option<Box<dyn Fn() -> bool + Send + Sync>>,
        interval: Duration,
    ) -> Self {
        Self {
            flag: AtomicBool::new(false),
            reason: AtomicU8::new(CancelReason::None.as_u8()),
            progress: AtomicUsize::new(0),
            probe,
            // u64::MAX 是「尚未探测过」的哨兵，保证第一次 poll() 必定探测：
            // 否则进程刚启动、now_us() 还是 0 时，第一次探测会被节流吞掉。
            last_probe_us: AtomicU64::new(u64::MAX),
            probe_interval_us: interval.as_micros().max(1) as u64,
        }
    }

    pub fn cancel(&self, reason: CancelReason) {
        // R6-1 P2：pump loop 可能同时收到信号 / timeout / 外部 token.cancel()。
        // 用 reason 的 CAS 保证「先到者赢」，且 reason 一定先于 flag 可见
        // （读端先 is_cancelled() 再 reason()，不会读到 None）。
        let r = match reason {
            CancelReason::None => CancelReason::User.as_u8(),
            other => other.as_u8(),
        };
        if self
            .reason
            .compare_exchange(
                CancelReason::None.as_u8(),
                r,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.flag.store(true, Ordering::Release);
        }
    }

    /// 细粒度检查点：单次 relaxed/acquire 原子读，无锁、无副作用。
    #[inline(always)]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    pub fn reason(&self) -> CancelReason {
        CancelReason::from_u8(self.reason.load(Ordering::Acquire))
    }

    /// core 在产出 `Cancelled` 之前上报进度（零额外开销：只发生在取消路径）。
    #[inline(always)]
    pub fn set_progress(&self, done: usize) {
        self.progress.store(done, Ordering::Relaxed);
    }

    pub fn progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    /// 粗粒度检查点：节流探测信号 + 读标志。
    ///
    /// `compare_exchange` 保证多线程下同一拍只有一个调用者执行 probe，
    /// 避免信号处理器被并发重复触发。
    pub fn poll(&self) -> bool {
        if let Some(probe) = &self.probe {
            let now = now_us();
            let last = self.last_probe_us.load(Ordering::Relaxed);
            let due = last == u64::MAX || now.saturating_sub(last) >= self.probe_interval_us;
            if due
                && self
                    .last_probe_us
                    .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                && probe()
            {
                self.cancel(CancelReason::Signal);
            }
        }
        self.is_cancelled()
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

// 编译期保证：后续阶段把令牌搬进 worker 线程时，类型层面必须已成立。
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CancelToken>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn reason_is_readable_after_cancel() {
        let t = CancelToken::new();
        assert!(!t.is_cancelled());
        assert_eq!(t.reason(), CancelReason::None);
        t.cancel(CancelReason::User);
        assert!(t.is_cancelled());
        assert_eq!(t.reason(), CancelReason::User);
    }

    #[test]
    fn cancel_only_when_probe_true() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let t = CancelToken::with_probe(
            Some(Box::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
                false
            })),
            Duration::ZERO,
        );
        for _ in 0..10 {
            assert!(!t.poll());
        }
        // 只保证至少探测过一次；具体次数取决于微秒分辨率，不作断言。
        assert!(calls.load(Ordering::Relaxed) >= 1);

        let t2 = CancelToken::with_probe(Some(Box::new(|| true)), Duration::ZERO);
        assert!(t2.poll());
        assert_eq!(t2.reason(), CancelReason::Signal);
    }

    #[test]
    fn probe_is_throttled() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let t = CancelToken::with_probe(
            Some(Box::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
                false
            })),
            Duration::from_secs(3600),
        );
        for _ in 0..1000 {
            assert!(!t.poll());
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn token_is_shareable_across_threads() {
        let t = Arc::new(CancelToken::new());
        let t2 = t.clone();
        std::thread::spawn(move || t2.cancel(CancelReason::Timeout))
            .join()
            .unwrap();
        assert!(t.is_cancelled());
        assert_eq!(t.reason(), CancelReason::Timeout);
    }
}

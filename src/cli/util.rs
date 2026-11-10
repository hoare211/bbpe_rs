// src/cli/util.rs
//
// 命令行公共工具：单位换算、内存预算、数值解析、人类可读输出。
//
// ⚠️ **单位换算是本项目的历史坑**：sweep 曾以 merges 数 K 为单位，而 train/cut
//    以总词表（256+K）为单位，同一份"100000"在两个工具里含义不同，文档被迫写了
//    三处警告。现统一为**总词表**口径，且换算只允许经过本文件的 `vocab_to_k`。

use std::io::Write;

pub type CliResult = Result<(), Box<dyn std::error::Error>>;

/// 初始化 tracing（只生效一次）。
///
/// 此前 **整个 CLI 从未初始化 subscriber**：只有 `bench` / `run_real_data` 与
/// Python 绑定做了初始化，于是 `trainer.rs` 里的阶段耗时、进度、GC 统计等
/// `info!` 在 `bbpe train` 下全部被丢弃 —— 既看不到训练进度，也拿不到性能分解，
/// `RUST_LOG` 环境变量同样无效。
///
/// 级别取 `RUST_LOG`（默认 warn，避免刷屏），输出到 stderr 以免污染 stdout
/// 的重定向产物（如 `bbpe encode` 的 id 列表）。
pub fn init_tracing_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let raw = std::env::var("RUST_LOG")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let level = if raw.contains("trace") {
            tracing::Level::TRACE
        } else if raw.contains("debug") {
            tracing::Level::DEBUG
        } else if raw.contains("info") {
            tracing::Level::INFO
        } else if raw.contains("error") {
            tracing::Level::ERROR
        } else {
            tracing::Level::WARN
        };
        let _ = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_target(false)
            .with_writer(std::io::stderr)
            .try_init();
    });
}

/// R6-1：CLI 的进程级 SIGINT → 取消令牌。
///
/// 原生 Rust 二进制的 SIGINT 默认直接终止进程，`TempFileGuard::drop` 不会
/// 运行 ⟹ 采样临时文件残留、进度条卡住。注册 handler 把信号变成
/// `CancelToken`，core 检查点返回 `Err(Cancelled)`，CLI 得以正常展开栈、
/// 由 RAII 清理临时文件后再退出。
///
/// `ctrlc::set_handler` 一个进程只能装一次，故用 `OnceLock` 缓存令牌；
/// 同一进程多次调用（分发器连续执行命令）共享同一个令牌。
pub fn sigint_cancel_token() -> std::sync::Arc<crate::cancel::CancelToken> {
    use std::sync::{Arc, OnceLock};

    static TOKEN: OnceLock<Arc<crate::cancel::CancelToken>> = OnceLock::new();
    TOKEN
        .get_or_init(|| {
            let token = Arc::new(crate::cancel::CancelToken::new());
            let t = token.clone();
            if let Err(e) = ctrlc::set_handler(move || {
                t.cancel(crate::cancel::CancelReason::Signal);
            }) {
                eprintln!("警告：无法注册 SIGINT handler（Ctrl+C 将直接终止进程）: {e}");
            }
            token
        })
        .clone()
}

/// 基础字节词表大小（byte-level BPE 的 0–255）
pub const BASE_VOCAB: usize = 256;

/// 经验内存系数：全内存精确训练的峰值 RSS ≈ `PEAK_MEM_FACTOR × 语料字节数`
pub const PEAK_MEM_FACTOR: f64 = 9.2;

/// 总词表 → merges 数 K。**唯一**的换算入口。
pub fn vocab_to_k(vocab: usize) -> usize {
    vocab.saturating_sub(BASE_VOCAB)
}

/// merges 数 K → 总词表
pub fn k_to_vocab(k: usize) -> usize {
    k + BASE_VOCAB
}

/// 启发式：看起来像"merges 数"的整千数字。
/// 总词表口径下 `256 + K` 几乎不可能是整千（100256 / 150256 …），
/// 故整千值基本可以判定为用户沿用了旧的 K 口径 ⟹ 给一次提示即可。
pub fn looks_like_k(v: usize) -> bool {
    v >= 1000 && v % 1000 == 0
}

/// 峰值 RSS（GB），读 /proc/self/status 的 VmHWM（Linux；其他平台返回 0）
pub fn peak_rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM"))
                .and_then(|l| l.split_whitespace().nth(1).map(|v| v.to_string()))
        })
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
        / 1_000_000.0
}

/// 可用内存（字节），读 /proc/meminfo 的 MemAvailable（Linux）
/// R6-1 P3 补遗：可用内存读取的三层回退。
///
/// ① `/proc/meminfo` 的 `MemAvailable`，缺失时回退 `MemTotal`；
/// ② 容器里 `/proc/meminfo` 显示的是**宿主机**内存，必须再取 cgroup 限额
///    （v2 `memory.max` / v1 `memory.limit_in_bytes`）并取两者较小值；
/// ③ 全部读不到才返回 `None`，由调用方显式告警（不再静默跳过预检）。
fn meminfo_bytes() -> Option<u64> {
    let parse_kb = |rest: &str| {
        rest.split_whitespace()
            .next()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|kb| kb * 1024)
    };
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut avail: Option<u64> = None;
    let mut total: Option<u64> = None;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail = parse_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_kb(rest);
        }
    }
    avail.or(total)
}

fn cgroup_limit_bytes() -> Option<u64> {
    const UNLIMITED_FLOOR: u64 = 1 << 60;
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ] {
        let Ok(s) = std::fs::read_to_string(path) else {
            continue;
        };
        let t = s.trim();
        if t == "max" {
            continue;
        }
        // v1 的 "unlimited" 是接近 2^63 的哨兵值，不能当真实上限
        if let Ok(v) = t.parse::<u64>() {
            if v < UNLIMITED_FLOOR {
                return Some(v);
            }
        }
    }
    None
}

/// 把「宿主机读数 / cgroup 限额」合成有效可用内存（**纯函数**，便于注入测试）。
/// 容器场景必须取较小值，否则按宿主机内存判定会在 cgroup OOM 时误放行。
pub fn effective_available_bytes(meminfo: Option<u64>, cgroup: Option<u64>) -> Option<u64> {
    match (meminfo, cgroup) {
        (Some(host), Some(cgroup)) => Some(host.min(cgroup)),
        (Some(host), None) => Some(host),
        (None, Some(cgroup)) => Some(cgroup),
        (None, None) => None,
    }
}

pub fn available_ram_bytes() -> Option<u64> {
    effective_available_bytes(meminfo_bytes(), cgroup_limit_bytes())
}

/// R4-1：OOM 预检判定（**纯函数**，阈值可注入）。
///
/// 原判定内联在 `cli/train.rs` 里、依赖 `/proc/meminfo` 实测，测试内无法
/// 稳定触发“超预算”分支（总不能真把机器内存吃满）。抽出后：
/// - 调用方传入实测的 `avail_bytes`（`None` = 读不到 MemAvailable，不拦）；
/// - 单测可注入任意阈值确定性覆盖边界（含恰好相等不拦）。
///
/// 语义不变：`train_bytes × PEAK_MEM_FACTOR > avail` 即超预算。该分支的
/// 铁律是**绝不删除任何输入文件**（R4-1 P0 的教训）。
pub fn exceeds_memory_budget(train_bytes: u64, avail_bytes: Option<u64>) -> bool {
    match avail_bytes {
        Some(avail) => train_bytes as f64 * PEAK_MEM_FACTOR > avail as f64,
        None => false,
    }
}

/// 人类可读字节数
pub fn human_bytes(b: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.2} {}", U[i])
    }
}

/// 输出文件自动建目录
pub fn ensure_parent(path: &str) -> std::io::Result<()> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    Ok(())
}

/// 判断两个路径是否指向同一个文件（R4-22：输入/输出别名防护）。
///
/// 先按规范化后的绝对路径比较；文件已存在时再用 `canonicalize`（解 symlink /
/// `..` 等）兜底。任一 `canonicalize` 失败（如输出尚未创建）时退化为
/// 规范化字符串比较 —— 目标是拦住"字面相同 / 同目录同文件名"的绝大多数误用，
/// 不追求覆盖所有 symlink 形态。
pub fn same_path(a: &str, b: &str) -> bool {
    let norm = |p: &str| {
        std::fs::canonicalize(p).unwrap_or_else(|_| {
            std::path::absolute(p).unwrap_or_else(|_| std::path::PathBuf::from(p))
        })
    };
    norm(a) == norm(b)
}

/// 解析逗号分隔的数值列表（`--vocabs 8448,16640` / `--at 1000,10000`）
pub fn parse_num_list<T>(s: &str, flag: &str) -> Result<Vec<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let out = s
        .split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| {
            x.parse::<T>()
                .map_err(|e| format!("{flag}: {e}（值 {x:?}）"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if out.is_empty() {
        return Err(format!("{flag}: 列表为空"));
    }
    Ok(out)
}

/// 临时文件的 RAII 守卫：离开作用域时（**含 `?` 提前返回**）自动删除。
///
/// n2 / n3：`cli/train` 的临时采样语料与 `cli/sweep` 的临时截断模型，此前只在
/// **正常结束**的路径上被删除；训练失败、参数非法等任何 `?` 提前返回都会把它们
/// 遗留在 `/tmp`（采样语料在真实语料下可达 GB 级）。改为守卫后清理不再依赖
/// 人工枚举退出路径。
pub struct TempFileGuard(Option<std::path::PathBuf>);

impl TempFileGuard {
    pub fn new(p: std::path::PathBuf) -> Self {
        Self(Some(p))
    }
    /// 放弃自动删除，把路径交还给调用方（例如"保留采样语料"的场景）
    pub fn keep(mut self) -> std::path::PathBuf {
        self.0.take().expect("TempFileGuard 只能 keep 一次")
    }
    pub fn path(&self) -> Option<&std::path::PathBuf> {
        self.0.as_ref()
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// 构造本进程专属的临时文件路径（调用方负责清理）
pub fn temp_path(prefix: &str, suffix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "{prefix}_{}_{}{suffix}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

/// 刷新并关闭 BufWriter（把 IO 错误显式暴露出来，而不是靠 drop 静默吞掉）
pub fn flush<W: Write>(w: &mut W) -> std::io::Result<()> {
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r6_1_available_ram_fallback_four_cases() {
        // 容器场景：cgroup 限额小于宿主机读数时必须取较小值
        assert_eq!(effective_available_bytes(Some(100), Some(40)), Some(40));
        assert_eq!(effective_available_bytes(Some(100), None), Some(100));
        assert_eq!(effective_available_bytes(None, Some(40)), Some(40));
        assert_eq!(effective_available_bytes(None, None), None);
        // cgroup 比宿主宽松时仍取宿主值
        assert_eq!(effective_available_bytes(Some(30), Some(40)), Some(30));
        // 本机实际读数必须可用（三层回退至少命中一层）
        assert!(available_ram_bytes().is_some());
    }

    #[test]
    fn unit_conversion_matches_train_sweep_cut() {
        assert_eq!(vocab_to_k(100_256), 100_000);
        assert_eq!(k_to_vocab(100_000), 100_256);
        // 防御：总词表小于 256 时饱和到 0，不允许下溢把 usize 打爆
        assert_eq!(vocab_to_k(0), 0);
        assert_eq!(vocab_to_k(256), 0);
    }

    // 旧 sweep 用 merges 数（K 口径）：整千取值应被识别为"可能是旧口径"。
    #[test]
    fn round_thousands_flagged_as_legacy_k() {
        // 旧 sweep 用 merges 数，典型取值都是整千 ⟹ 应触发提示
        assert!(looks_like_k(100_000));
        assert!(looks_like_k(150_000));
        assert!(looks_like_k(8_000));
        // 总词表口径下 256+K 几乎不可能是整千
        assert!(!looks_like_k(100_256));
        assert!(!looks_like_k(150_256));
        assert!(!looks_like_k(8_448));
    }

    #[test]
    fn numeric_list_parsing() {
        assert_eq!(
            parse_num_list::<usize>("8448, 16640,100256", "--vocabs").unwrap(),
            vec![8448, 16640, 100256]
        );
        assert!(parse_num_list::<usize>("", "--vocabs").is_err());
        assert!(parse_num_list::<usize>("100,abc", "--vocabs").is_err());
    }

    #[test]
    fn human_readable_bytes() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024), "1.00 KB");
        assert_eq!(human_bytes(50 << 20), "50.00 MB");
    }

    // R4-1：预检判定抽为纯函数后，阈值可注入 ⟹ 边界可确定性覆盖。
    #[test]
    fn r4_1_oom_precheck_boundaries() {
        // 读不到 MemAvailable ⟹ 不拦（保持旧行为：无法估算时不误报）
        assert!(!exceeds_memory_budget(u64::MAX, None));
        // 1 GB 语料 × 9.2 ≈ 9.2 GB > 1 GB 可用 ⟹ 超预算
        assert!(exceeds_memory_budget(1 << 30, Some(1 << 30)));
        // 充裕内存 ⟹ 不拦
        assert!(!exceeds_memory_budget(1 << 20, Some(1 << 30)));
        // 阈值两侧（×9.2 ≈ 9.2 MB）：明确裕量，不依赖浮点精确相等
        // （PEAK_MEM_FACTOR=9.2 无法用 f64 精确表示，恰好相等的断言会脆）
        let train = 1_000_000u64;
        assert!(!exceeds_memory_budget(train, Some(20_000_000)));
        assert!(exceeds_memory_budget(train, Some(9_000_000)));
        // 空语料 ⟹ 永不拦
        assert!(!exceeds_memory_budget(0, Some(0)));
    }
}

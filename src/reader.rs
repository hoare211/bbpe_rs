// src/reader.rs
use crate::error::BbpeError;
use crate::tokenizer::CorpusMeta;
use sonic_rs::{Deserialize, JsonValueTrait};
use std::borrow::Cow;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[inline(always)]
pub(crate) fn strip_newline(buf: &[u8]) -> &[u8] {
    let mut len = buf.len();
    while len > 0 && (buf[len - 1] == b'\n' || buf[len - 1] == b'\r') {
        len -= 1;
    }
    &buf[..len]
}

// =========================================================
// 输入侧计数上报（规范 §6.2 / §6.3 / §6.6）
// =========================================================

/// 语料读取过程的**静默行为**计数器。
///
/// 背景（规范 §6.2/§6.3）：非法 UTF-8 被 `from_utf8_lossy` 静默替换为 U+FFFD、
/// JSONL 坏行被静默 `continue` 跳过——这些行为本期**保持不变**（ADR-5），
/// 但必须**可见**（G4 可观测），否则使用者无从得知语料被改动/丢弃了多少。
///
/// 计数需跨 `par_iter` 线程聚合，故内部全为原子量；通过 [`CorpusReader::iter_lines_with_stats`]
/// 传入，迭代结束后读取快照。
#[derive(Debug, Default)]
pub struct CorpusStats {
    /// 物理行数（含空行与坏行）
    pub total_lines: AtomicU64,
    /// 读入的原始字节数（含行尾换行符）⟹ 与文件字节数一致
    pub total_bytes: AtomicU64,
    /// 被静默跳过的行数（逐 reader 语义见 §6.3）
    pub skipped_lines: AtomicU64,
    /// 非法 UTF-8 被替换为 U+FFFD 的次数（§6.2）
    pub replacement_chars: AtomicU64,
    /// R4-7：读取中途发生的 IO 错误次数（错误后迭代即终止）。
    ///
    /// 存在意义：原实现把 IO 错误当 EOF（打印告警后 `return None`），
    /// 于是 NFS/坏块/句柄错误会让**整个剩余语料静默消失**，而
    /// `skipped_lines` 只能 +1，完全无法表达"后面全都没读"。
    /// 现在严格模式直接向上传播错误；该计数器让非严格模式（兼容旧行为）
    /// 至少在 meta 里留下痕迹。
    pub io_errors: AtomicU64,
}

impl CorpusStats {
    /// 取 (total_lines, total_bytes, skipped_lines, replacement_chars) 快照
    pub fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.total_lines.load(Ordering::Relaxed),
            self.total_bytes.load(Ordering::Relaxed),
            self.skipped_lines.load(Ordering::Relaxed),
            self.replacement_chars.load(Ordering::Relaxed),
        )
    }

    /// R4-7：读取中途 IO 错误次数（>0 表示语料被**截断**，不是读完）
    pub fn io_errors(&self) -> u64 {
        self.io_errors.load(Ordering::Relaxed)
    }

    /// 按 §6.3 语义构造 `meta.corpus` 快照
    pub fn snapshot_meta(&self, reader: &str) -> CorpusMeta {
        let (total_lines, total_bytes, skipped_lines, replacement_chars) = self.snapshot();
        CorpusMeta {
            reader: reader.to_string(),
            total_lines,
            total_bytes,
            skipped_lines,
            replacement_chars,
            io_errors: self.io_errors(),
        }
    }
}

type SharedStats = Option<Arc<CorpusStats>>;

#[inline]
fn count_line(stats: &SharedStats, raw_len: usize) {
    if let Some(s) = stats {
        s.total_lines.fetch_add(1, Ordering::Relaxed);
        s.total_bytes.fetch_add(raw_len as u64, Ordering::Relaxed);
    }
}

#[inline]
fn count_skipped(stats: &SharedStats) {
    if let Some(s) = stats {
        s.skipped_lines.fetch_add(1, Ordering::Relaxed);
    }
}

/// R4-7：IO 错误计数（区分"读完"与"读炸了"）
#[inline]
fn count_io_error(stats: &SharedStats) {
    if let Some(s) = stats {
        s.io_errors.fetch_add(1, Ordering::Relaxed);
    }
}

/// 统计 `from_utf8_lossy` 引入的 U+FFFD 个数。
/// 注：无法区分"原文本就含 U+FFFD"，故该值是上界——但足以让静默替换**可见**。
#[inline]
fn note_replacement(stats: &SharedStats, s: &str) {
    if let Some(st) = stats {
        let n = s.chars().filter(|c| *c == '\u{FFFD}').count() as u64;
        if n > 0 {
            st.replacement_chars.fetch_add(n, Ordering::Relaxed);
        }
    }
}

// ====== 静态格式定义 (保留了你的极速反序列化模型) ======
#[derive(Deserialize)]
struct Format1<'a> {
    #[serde(borrow)]
    text: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct Message<'a> {
    #[serde(borrow)]
    content: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct FormatConversations<'a> {
    #[serde(borrow)]
    conversations: Option<Vec<Message<'a>>>,
}

/// R6-1 P1：`Clone` 使绑定层能在持 GIL 时把 reader 变成 owned 值搬进
/// worker 线程。字段全是 `String`/`Vec`，clone 成本极低，且不改变读取语义。
#[derive(Clone)]
pub enum CorpusReader {
    Text {
        filepath: String,
    },
    JsonlStaticText {
        filepath: String,
    },
    JsonlStaticConversations {
        filepath: String,
        sep: String,
    },

    // 【核心重构】：保存了提取路径和类型的动态配置
    JsonlDynamic {
        filepath: String,
        // 存储 (路径, 类型)，不再需要列名，因为数据全部喂给 BPE
        schema_rules: Vec<(Vec<String>, String)>,
    },
}

impl CorpusReader {
    /// R6-1 P3：训练前取得物理语料路径（采样输入 / OOM 预检需要）。
    /// 四种变体都只读一个文件，故可统一取出。
    pub fn filepath(&self) -> &str {
        match self {
            Self::Text { filepath }
            | Self::JsonlStaticText { filepath }
            | Self::JsonlStaticConversations { filepath, .. }
            | Self::JsonlDynamic { filepath, .. } => filepath,
        }
    }

    /// 训练前的语料字节数。`CorpusStats::total_bytes` 是**训练中累计**，
    /// 训练前恒为 0，因此预检必须绕过 reader 直接查 metadata。
    pub fn file_size(&self) -> u64 {
        std::fs::metadata(self.filepath())
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// 保持 reader 变体（及其格式/schema 配置）不变，只替换语料路径。
    /// 采样按物理行复制，产物仍是同格式文件，不能退化成 `Text` 变体。
    pub fn with_filepath(&self, filepath: String) -> Self {
        let mut r = self.clone();
        match &mut r {
            Self::Text { filepath: p }
            | Self::JsonlStaticText { filepath: p }
            | Self::JsonlStaticConversations { filepath: p, .. }
            | Self::JsonlDynamic { filepath: p, .. } => *p = filepath,
        }
        r
    }

    /// reader 类型标识，写入 `meta.corpus.reader`（规范 §7.1）
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Text { .. } => "text",
            Self::JsonlStaticText { .. } => "jsonl:text",
            Self::JsonlStaticConversations { .. } => "jsonl:conversations",
            Self::JsonlDynamic { .. } => "jsonl:dynamic",
        }
    }

    /// 原始接口：**行为与计数均不变**（不传计数器 ⟹ 零开销）。
    pub fn iter_lines(&self) -> Result<Box<dyn Iterator<Item = String>>, BbpeError> {
        // Result / Option 都实现 IntoIterator ⟹ 非严格模式的 Err 永不出
        // （已就地告警），flatten 直接过滤并解包
        Ok(Box::new(self.iter_lines_impl(None, false)?.flatten()))
    }

    /// 带计数上报的迭代（规范 §6.6）。语义与 [`Self::iter_lines`] 完全一致，
    /// 仅额外累加 `total_lines` / `total_bytes` / `skipped_lines` / `replacement_chars`。
    pub fn iter_lines_with_stats(
        &self,
        stats: Arc<CorpusStats>,
    ) -> Result<Box<dyn Iterator<Item = String>>, BbpeError> {
        Ok(Box::new(
            self.iter_lines_impl(Some(stats), false)?.flatten(),
        ))
    }

    /// R4-7：**严格**迭代 —— IO 错误不再被当作 EOF。
    ///
    /// 原实现在 `read_until` 出错时打印告警并 `return None`：NFS/磁盘坏块/
    /// 句柄错误会让后续语料**静默消失**，训练照常完成并保存"截断模型"，
    /// 而 `skipped_lines += 1` 无法表达"整个剩余文件未读"。
    /// 本方法把 IO 错误作为 `Err` 产出，调用方（训练器等）必须显式处理；
    /// 迭代在首个 IO 错误后终止（后续读取无意义）。
    pub fn iter_lines_checked(
        &self,
        stats: Option<Arc<CorpusStats>>,
    ) -> Result<Box<dyn Iterator<Item = Result<String, BbpeError>>>, BbpeError> {
        self.iter_lines_impl(stats, true)
    }

    // 你的静态读取方法我直接简化在这，核心看后面的 Dynamic
    //
    // R4-7：`strict` 决定 IO 错误的去向：
    //   - `false`（兼容旧行为）：告警 + 计数 + 终止迭代（Result 通道永不出 Err）；
    //   - `true`：产出 `Err(BbpeError::IoError)` 并终止迭代。
    fn iter_lines_impl(
        &self,
        stats: SharedStats,
        strict: bool,
    ) -> Result<Box<dyn Iterator<Item = Result<String, BbpeError>>>, BbpeError> {
        match self {
            Self::Text { filepath } => {
                let file = File::open(filepath)?;
                let mut reader = BufReader::new(file);
                let mut buf = Vec::new();
                let mut io_failed = false;
                let iter = std::iter::from_fn(move || {
                    // R4-7：strict 下首个 IO 错误后必须终止（from_fn 不会因
                    // Some(Err) 自动停摆，不加闩会反复重试、持续产出 Err）
                    if io_failed {
                        return None;
                    }
                    loop {
                        buf.clear();
                        match reader.read_until(b'\n', &mut buf) {
                            Ok(0) => return None,
                            Ok(_) => {
                                count_line(&stats, buf.len());
                                let slice = strip_newline(&buf);
                                if !slice.is_empty() {
                                    let line = String::from_utf8_lossy(slice).into_owned();
                                    note_replacement(&stats, &line);
                                    return Some(Ok(line));
                                }
                            }
                            // M4 / R4-7：IO 错误 ≠ EOF。strict 模式下作为 Err 产出
                            // （调用方显式失败）；非严格模式保持旧行为：告警 + 计数 + 终止。
                            Err(e) => {
                                eprintln!(
                                    "⚠️  bbpe_rs: 语料读取发生 IO 错误，迭代提前终止（其后语料未被统计）: {e}"
                                );
                                count_io_error(&stats);
                                if strict {
                                    io_failed = true;
                                    return Some(Err(BbpeError::IoError(e)));
                                }
                                count_skipped(&stats);
                                return None;
                            }
                        }
                    }
                });
                Ok(Box::new(iter))
            }

            Self::JsonlStaticText { filepath } => {
                let file = File::open(filepath)?;
                let mut reader = BufReader::new(file);
                let mut buf = Vec::new();
                let mut io_failed = false;
                let iter = std::iter::from_fn(move || {
                    // R4-7：strict 下首个 IO 错误后必须终止（from_fn 不会因
                    // Some(Err) 自动停摆，不加闩会反复重试、持续产出 Err）
                    if io_failed {
                        return None;
                    }
                    loop {
                        buf.clear();
                        match reader.read_until(b'\n', &mut buf) {
                            Ok(0) => return None,
                            Ok(_) => {
                                count_line(&stats, buf.len());
                                let slice = strip_newline(&buf);
                                if slice.is_empty() {
                                    continue;
                                }
                                if let Ok(record) = sonic_rs::from_slice::<Format1>(slice) {
                                    if let Some(text) = record.text {
                                        let line = text.into_owned();
                                        note_replacement(&stats, &line);
                                        return Some(Ok(line));
                                    }
                                }
                                // §6.3：行非空 且（JSON 解析失败 或 text 缺失）⟹ skipped +1
                                count_skipped(&stats);
                            }
                            // M4 / R4-7：IO 错误 ≠ EOF。strict 模式下作为 Err 产出
                            // （调用方显式失败）；非严格模式保持旧行为：告警 + 计数 + 终止。
                            Err(e) => {
                                eprintln!(
                                    "⚠️  bbpe_rs: 语料读取发生 IO 错误，迭代提前终止（其后语料未被统计）: {e}"
                                );
                                count_io_error(&stats);
                                if strict {
                                    io_failed = true;
                                    return Some(Err(BbpeError::IoError(e)));
                                }
                                count_skipped(&stats);
                                return None;
                            }
                        }
                    }
                });
                Ok(Box::new(iter))
            }

            Self::JsonlStaticConversations { filepath, sep } => {
                let file = File::open(filepath)?;
                let mut reader = BufReader::new(file);
                let mut buf = Vec::new();
                let sep_clone = sep.clone();
                let mut io_failed = false;
                let iter = std::iter::from_fn(move || {
                    // R4-7：strict 下首个 IO 错误后必须终止（from_fn 不会因
                    // Some(Err) 自动停摆，不加闩会反复重试、持续产出 Err）
                    if io_failed {
                        return None;
                    }
                    loop {
                        buf.clear();
                        match reader.read_until(b'\n', &mut buf) {
                            Ok(0) => return None,
                            Ok(_) => {
                                count_line(&stats, buf.len());
                                let slice = strip_newline(&buf);
                                if slice.is_empty() {
                                    continue;
                                }
                                if let Ok(record) =
                                    sonic_rs::from_slice::<FormatConversations>(slice)
                                {
                                    if let Some(convs) = record.conversations {
                                        let joined = convs
                                            .iter()
                                            .filter_map(|msg| msg.content.as_deref())
                                            .collect::<Vec<_>>()
                                            .join(&sep_clone);
                                        if !joined.is_empty() {
                                            note_replacement(&stats, &joined);
                                            return Some(Ok(joined));
                                        }
                                    }
                                }
                                // §6.3：解析失败 / conversations 缺失 / 拼接结果为空 ⟹ skipped +1
                                count_skipped(&stats);
                            }
                            // M4 / R4-7：IO 错误 ≠ EOF。strict 模式下作为 Err 产出
                            // （调用方显式失败）；非严格模式保持旧行为：告警 + 计数 + 终止。
                            Err(e) => {
                                eprintln!(
                                    "⚠️  bbpe_rs: 语料读取发生 IO 错误，迭代提前终止（其后语料未被统计）: {e}"
                                );
                                count_io_error(&stats);
                                if strict {
                                    io_failed = true;
                                    return Some(Err(BbpeError::IoError(e)));
                                }
                                count_skipped(&stats);
                                return None;
                            }
                        }
                    }
                });
                Ok(Box::new(iter))
            }

            // =====================================================================
            // 完美复活！强大的动态多维度提取引擎，专为 BPE 提供海量养料
            // =====================================================================
            Self::JsonlDynamic {
                filepath,
                schema_rules,
            } => {
                let file = File::open(filepath)?;
                let mut reader = BufReader::new(file);
                let mut buf = Vec::new();
                let rules = schema_rules.clone();

                // 内部字符串缓冲池：因为单行 JSON 可能会被解析出多个文本块。
                // m5：用 VecDeque 而非 Vec —— 原实现 `Vec::pop()` 是后进先出，会把
                // 同一 JSONL 行内多个规则产出的文本**顺序颠倒**。字节级 BPE 的相邻
                // pair 统计虽不受影响，但"产出顺序 == 规则声明顺序"这一直觉会失效，
                // 给调试与多字段语义带来困扰。改 pop_front() 后严格保序。
                let mut string_buffer: VecDeque<String> = VecDeque::new();

                let mut io_failed = false;
                let iter = std::iter::from_fn(move || {
                    // R4-7：strict 下首个 IO 错误后必须终止（from_fn 不会因
                    // Some(Err) 自动停摆，不加闩会反复重试、持续产出 Err）
                    if io_failed {
                        return None;
                    }
                    loop {
                        // 优先排空缓冲池给 BPE 引擎（FIFO，保持规则声明顺序）
                        if let Some(s) = string_buffer.pop_front() {
                            return Some(Ok(s));
                        }

                        buf.clear();
                        match reader.read_until(b'\n', &mut buf) {
                            Ok(0) => return None,
                            Ok(_) => {
                                count_line(&stats, buf.len());
                                let slice = strip_newline(&buf);
                                if slice.is_empty() {
                                    continue;
                                }

                                let produced_before = string_buffer.len();
                                for (pointers, ty) in &rules {
                                    let ptrs: Vec<&str> =
                                        pointers.iter().map(|s| s.as_str()).collect();
                                    let lazy_val = sonic_rs::get_from_slice(slice, ptrs.as_slice());

                                    // 【精髓复现】：长长的动态判断逻辑！
                                    // 无论用户的数据长什么样，我们全能给它榨出来转成字符串喂给 Tokenizer！
                                    match ty.as_str() {
                                        // 【必须解析区】：需要脱除 JSON 双引号和处理转义符
                                        "str" | "string" => {
                                            if let Some(s) = lazy_val
                                                .ok()
                                                .and_then(|lv| lv.as_str().map(|s| s.to_string()))
                                            {
                                                string_buffer.push_back(s.to_string());
                                            }
                                        }
                                        "list_str" => {
                                            if let Ok(lv) = lazy_val {
                                                if lv.is_array() {
                                                    // 解开 JSON 数组，获取纯净文本
                                                    if let Ok(arr) =
                                                        sonic_rs::from_slice::<Vec<String>>(
                                                            lv.as_raw_str().as_bytes(),
                                                        )
                                                    {
                                                        string_buffer.extend(arr);
                                                    }
                                                }
                                            }
                                        }

                                        // 【极限零开销区】：完全摒弃类型转换！
                                        // 无论是 int, float, bool 还是复杂的嵌套 json，
                                        // 我们统统利用 sonic-rs 的 as_raw_str() 原地截取底层字节切片，零解析开销！
                                        "int" | "float" | "bool" | "raw_json" => {
                                            if let Ok(lv) = lazy_val {
                                                // 直接截取 JSON 源码中的切片 (例如 "19.9" 或者 "true")
                                                string_buffer
                                                    .push_back(lv.as_raw_str().to_string());
                                            }
                                        }

                                        // 对于 list_int，如果用户想要提取 [1, 2]，其实作为 raw_json 提取即可
                                        // 如果用户非要提取出单独的 1 和 2 给 Tokenizer 学，依然可以通过反序列化解开：
                                        "list_int" => {
                                            if let Ok(lv) = lazy_val {
                                                if lv.is_array() {
                                                    if let Ok(arr) = sonic_rs::from_slice::<Vec<i64>>(
                                                        lv.as_raw_str().as_bytes(),
                                                    ) {
                                                        for num in arr {
                                                            string_buffer
                                                                .push_back(num.to_string());
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        _ => continue,
                                    }
                                }
                                // §6.3：行非空 且 未产出任何文本块 ⟹ skipped +1
                                // （行内部分字段失败而其他成功 ⟹ 不计，按行粒度）
                                if string_buffer.len() == produced_before {
                                    count_skipped(&stats);
                                }
                            }
                            // M4 / R4-7：IO 错误 ≠ EOF。原实现把两者走同一条路径
                            // （`Err(_) => return None`），网络文件系统 / 坏块 / 权限变更
                            // 等场景下语料会被**静默截断**，且 meta.corpus.total_bytes
                            // 随之偏小而完全没有信号。R5 轮补齐：动态 reader 与其余三种
                            // 同口径 —— strict 下产出 `Err(BbpeError::IoError)`（训练显式
                            // 失败，Python `JsonlFileReader.dynamic` 亦可感知）；非严格
                            // 模式保持旧行为（告警 + 计数 + 终止），且计入 io_errors
                            // 而非 skipped_lines（后者无法表达"整个剩余文件未读"）。
                            Err(e) => {
                                eprintln!(
                                    "⚠️  bbpe_rs: 语料读取发生 IO 错误，迭代提前终止（其后语料未被统计）: {e}"
                                );
                                count_io_error(&stats);
                                if strict {
                                    io_failed = true;
                                    return Some(Err(BbpeError::IoError(e)));
                                }
                                count_skipped(&stats);
                                return None;
                            }
                        }
                    }
                });
                Ok(Box::new(iter))
            }
        }
    }
}

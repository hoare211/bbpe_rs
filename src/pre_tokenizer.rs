use crate::error::BbpeError;
use fancy_regex::Regex;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

static REGEX_CACHE: OnceLock<RwLock<HashMap<String, Regex>>> = OnceLock::new();

// // 设定一个 Chunk 的最大长度（字符数）。如果超过这个长度，强制切断。
// // 这个阈值既保护了内存不被大 Chunk 挤爆，又保留了正常的词语合并！
// const MAX_CHUNK_LENGTH_CHARS: usize = 256;

// 我们不再按“字符个数”限制，而是按“字节长度”限制。
// 512 字节大约相当于 170 个汉字，足够 BPE 发现任何常见中文词组。
/// 默认单 chunk 字节上限。
///
/// m4：原先它是本文件内的私有常量，而 `RuntimeConfig::max_chunk_bytes` 只是
/// 被写进 `meta.train.max_chunk_bytes` 却**不参与任何切分** —— 一旦用户调节该
/// 配置，meta 记录值与实际切分行为就会不符（契约记录必须等于事实）。
/// 现在它是默认值，`PreTokenizer` 实例可携带自己的上限。
pub const DEFAULT_MAX_CHUNK_BYTES: usize = 512;

// 我们内置的工业级正则字典
/// 项目自研默认正则：参考 cl100k_base 的多行 (?x) 写法，覆盖缩写 / 字母 / 数字 / 标点 / 空白。
pub const PATTERN_CHINESE_OPTIMIZED: &str = r"(?x)
                '(?i:[sdmt]|ll|ve|re)          # 1. 英文缩写
              | \p{Han}+                       # 2. 汉字连缀匹配 (加上 + 号)
              | \ ? [\p{L}--\p{Han}]+          # 3. 纯字母匹配
              | \ ? \p{N}+                     # 4. 数字
              | \ ? [^\s\p{L}\p{N}]+           # 5. 标点符号
              | \s+                            # 6. 换行与纯空格
            ";

// pub const PATTERN_CHINESE_OPTIMIZED: &str = r"(?x)
//                 '(?i:[sdmt]|ll|ve|re)          # 1. 英文缩写
//               | \ ? \p{L}+                     # 2. 字母与多国语言文字 (带或不带前导空格)
//               | \ ? \p{N}+                     # 3. 数字
//               | \ ? [^\s\p{L}\p{N}]+           # 4. 标点符号独立
//               | \s+                            # 5. 换行与纯空格
//             ";

pub const PATTERN_GPT4: &str = r"(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
pub const PATTERN_LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// 全项目唯一的默认 pre-tokenizer pattern。改这里一处即可，所有 fallback 共用。
pub const DEFAULT_PATTERN: &str = PATTERN_CHINESE_OPTIMIZED;

/// A2：构造"未覆盖间隙"的报错（带内容样本，便于直接定位问题 pattern）。
///
/// 样本截断到 60 字节 —— 足够看出是哪类字符没被匹配，又不会把整行语料
/// 灌进错误消息里（语料可能是敏感内容，也可能很长）。
struct PreTokenizeError;

impl PreTokenizeError {
    fn gap(missing: &str, at: usize) -> BbpeError {
        const SAMPLE: usize = 60;
        let n = missing.len();
        // R4-3：截断点必须落到 UTF-8 字符边界。`&missing[..60]` 在第 60 字节
        // 落在多字节字符内部时会直接 panic —— 本应返回 PreTokenizeError 的
        // 路径变成 panic，与"错误映射为异常、不泄漏 panic"的契约冲突。
        let cut = floor_char_boundary(missing, SAMPLE.min(n));
        let sample = if n <= SAMPLE {
            missing.to_string()
        } else {
            format!("{}…（共 {n} 字节）", &missing[..cut])
        };
        BbpeError::PreTokenize(format!(
            "预分词正则**未覆盖**字节 {at}..{}（{n} 字节丢失）：{sample:?}。\n\
             预分词正则必须覆盖输入的**每一个字符** —— `find_iter` 只产出匹配，\
             匹配之间的间隙会被**静默丢弃**（既不计入 skipped_lines，也不会有任何告警）。\n\
             处置：修正 pattern 使其全函数；或改用内置的 DEFAULT_PATTERN / \
             PATTERN_GPT4 / PATTERN_LLAMA3（均已实测为全函数）。",
            at + n
        ))
    }
}

/// A2：预分词失败的**保证可见**告警通道。
///
/// 为什么不能只靠 `tracing::error!`：Python 进程默认**没有**安装 tracing
/// subscriber（需显式 `init_rust_logger()`），CLI 侧也只在个别 bin 里安装。
/// 于是 `tracing` 的错误在最重要的两条使用路径上都会消失 —— 而这正是
/// `reader.rs` 的 M4 修复选择 `eprintln!` 的同一理由（见该文件注释）。
///
/// 这里同时走两条通道，保证"绝不静默"：
/// 1. `eprintln!` —— 无条件写 stderr，任何环境都可见；
/// 2. `tracing::error!` —— 装了 subscriber 的场景可结构化采集。
///
/// 影响面提示：一旦触发，说明该行文本**未被完整预分词**：
/// - 训练路径（`tokenize_with`）通常同时有 `trainer.rs` 的硬失败守卫；
/// - 推理路径（`encode`）会导致 `decode(encode(text)) != text`。
///
/// R4-4：可见性放宽到 crate 内 —— `tokenizer.rs` 的非严格 encode 路径
/// 需要复用它报警（与训练路径的硬失败并存）。
pub(crate) fn report_pretokenize_error(e: &BbpeError) {
    eprintln!(
        "❌ bbpe_rs: 预分词失败，该行文本未被完整切分（已匹配部分有效，但**整行不完整**）: {e}"
    );
    tracing::error!(error = %e, "预分词失败（整行文本不完整）");
}

/// 把字节下标 `i` 向下取到 `s` 的 UTF-8 字符边界（R4-3）。
///
/// `str::is_char_boundary` 在非边界下标上返回 false，而 `&s[..i]` 会 panic；
/// 错误消息构造、样本截断等所有"按字节数截断字符串"的场景都必须先过这里。
pub(crate) fn floor_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[derive(Clone)]
pub struct PreTokenizer {
    regex: Regex, // <-- 将正则编译后保存在实例中
    /// R4-23：构造时的 pattern 原文（fancy_regex 不提供取回源串的 API）
    pattern: String,
    /// m4：单 chunk 字节上限（真正参与切分）。默认 [`DEFAULT_MAX_CHUNK_BYTES`]。
    max_chunk_bytes: usize,
}

impl PreTokenizer {
    /// 接收一个传入的正则字符串并将其编译为极其高效的引擎。
    /// chunk 上限取默认值 [`DEFAULT_MAX_CHUNK_BYTES`]。
    pub fn new(pattern: &str) -> Result<Self, BbpeError> {
        Self::with_max_chunk_bytes(pattern, DEFAULT_MAX_CHUNK_BYTES)
    }

    /// 同上，但显式指定单 chunk 字节上限（m4：让 `RuntimeConfig::max_chunk_bytes`
    /// 真正参与切分，使 meta 记录值与行为一致）。
    ///
    /// `max_chunk_bytes = 0` 会被视为"不切分"（交由调用方保证 chunk 可控），
    /// 但通常应保持 ≥ 64 以避免产生超长 chunk 撑爆内存。
    pub fn with_max_chunk_bytes(pattern: &str, max_chunk_bytes: usize) -> Result<Self, BbpeError> {
        // 获取或初始化全局缓存池
        let cache = REGEX_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

        // 【极速路径】：只读锁，如果该 Pattern 已经编译过，直接克隆引用
        if let Ok(guard) = cache.read() {
            if let Some(cached_regex) = guard.get(pattern) {
                return Ok(Self {
                    regex: cached_regex.clone(),
                    pattern: pattern.to_string(),
                    max_chunk_bytes,
                });
            }
        }

        // 【慢速路径】：编译正则（仅在系统启动 / 遇到新 Pattern 时执行一次）
        let regex = Regex::new(pattern)
            .map_err(|e| BbpeError::RegexError(regex::Error::Syntax(e.to_string())))?;

        // 写入全局缓存
        if let Ok(mut guard) = cache.write() {
            guard.insert(pattern.to_string(), regex.clone());
        }

        Ok(Self {
            regex,
            pattern: pattern.to_string(),
            max_chunk_bytes,
        })
    }

    /// 当前生效的单 chunk 字节上限（m4：供 meta 回读与测试断言）
    pub fn max_chunk_bytes(&self) -> usize {
        self.max_chunk_bytes
    }

    /// 本实例的预分词正则（R4-23：供 `BBPETrainer::set_pre_tokenizer`
    /// 把实际预分词器同步回 state/config，避免三者静默分叉）。
    pub fn pattern(&self) -> &str {
        // regex 不提供取回源串的 API，故由构造时一并保存
        &self.pattern
    }

    /// 预分词，并对**每一处正则错误与未覆盖间隙**显式返回（训练路径必须用它）。
    ///
    /// # A2：真正的机制是"间隙丢失"，不是"错误被吞"
    ///
    /// 初版评审把问题归因为 `if let Ok(mat) = mat_result` 吞掉了 `Err`。实测
    /// 复现后**该归因不成立**：fancy_regex 0.18 对灾难性回溯 pattern 返回的是
    /// `Ok`（零匹配）而非 `Err` —— 实测 `(a+)+$` 配 5000 个 `a` + `b` 仍返回
    /// `Ok`，只是 `chunks=0`。`clippy` 那条 `unnecessary if let` 告警是对的
    /// （`Err` 分支确实该处理），但那只是次要问题。
    ///
    /// 真正的机制是 pre-tokenization 的经典陷阱：**`find_iter` 只产出匹配，
    /// 匹配之间的"间隙"（gap）永远不会进入回调**。因此只要 pattern 不是
    /// **全函数**（覆盖输入的每一个字符），未匹配部分就会被**静默丢弃**。
    ///
    /// 实测差距（`\s+` 这种"看起来没问题"的 pattern）：
    /// ```text
    ///   "你好世界"   → ""        (丢 12 字节)
    ///   "a😀b"       → ""        (丢 6 字节)
    ///   "123 456"    → " "       (丢 6 字节)
    ///   "!?.,;"      → ""        (丢 5 字节)
    /// ```
    /// 训练侧后果最重：整行语料被丢弃且**不计入 `skipped_lines`**（该计数在
    /// `reader.rs`，看不到 pre-tokenizer 层的问题）⟹ `meta.corpus` 显示一切正常，
    /// 模型静默缺数据。
    ///
    /// 内置的三个 pattern（[`DEFAULT_PATTERN`] / [`PATTERN_GPT4`] /
    /// [`PATTERN_LLAMA3`]）实测均为**全函数**（16/16 边界用例完整覆盖，含
    /// emoji、孤立延续字节、BOM、RTL 标记、组合记号、零宽空格），故默认配置下
    /// 不会触发本检查。但自定义 pattern 是**公开 API**
    /// （`TrainerConfig::with_pre_tokenizer_pattern`、`PatternFlavor::Custom`，
    /// 以及**从模型文件读入的 `pre_tokenizer_pattern`**），一份覆盖不全的
    /// pattern 就能让整个语料集静默消失。
    ///
    /// 因此本方法把"未覆盖"当作**硬错误**：报出缺失的字节区间与内容样本，
    /// 让使用者能直接定位并修好 pattern。`Err` 时已回调的 chunk 仍属有效产出，
    /// 但语义是"本次切分**不完整**"，调用方应丢弃本轮结果而非当作完整语料。
    pub fn tokenize_with_checked<'a, F>(&self, line: &'a str, mut f: F) -> Result<(), BbpeError>
    where
        F: FnMut(&'a [u8], usize),
    {
        // m4：切分上限取实例值（此前是文件内常量，导致 RuntimeConfig 的同名
        // 配置形同虚设）。max_chunk_bytes = 0 表示不切分。
        let max_chunk = if self.max_chunk_bytes == 0 {
            usize::MAX
        } else {
            self.max_chunk_bytes
        };

        // A2：逐次比对"上一个匹配的结束"与"下一个匹配的开始"，
        // 二者不等即说明中间有**未被预分词**的间隙。
        let mut covered_to = 0usize;
        for mat_result in self.regex.find_iter(line) {
            // 次要但仍需处理：运行期错误（如回溯超限）此前被静默丢弃
            let mat = mat_result.map_err(|e| {
                BbpeError::PreTokenize(format!(
                    "预分词正则运行期失败（通常回溯超限；请简化 pattern 或缩短输入行）: {e}"
                ))
            })?;
            let (m_start, m_end) = (mat.start(), mat.end());

            if m_start < covered_to {
                return Err(BbpeError::PreTokenize(format!(
                    "预分词正则产出了**重叠**匹配（上一个匹配结束于 {covered_to}，下一个始于 {m_start}）—— \
                     请检查 pattern 是否可匹配空串或存在自重叠"
                )));
            }
            if m_start > covered_to {
                return Err(PreTokenizeError::gap(
                    &line[covered_to..m_start],
                    covered_to,
                ));
            }

            let chunk = &line[m_start..m_end];
            covered_to = m_end;

            if chunk.len() <= max_chunk {
                f(chunk.as_bytes(), m_start);
                continue;
            }

            // 超长 chunk 按 UTF-8 字符边界强制切分
            let bytes = chunk.as_bytes();
            let mut start = 0;
            while start < chunk.len() {
                let mut end = start + max_chunk;
                if end > chunk.len() {
                    end = chunk.len();
                } else {
                    // R4-2：`max_chunk_bytes` 取 1/2 这类极小值时，`end` 会一路
                    // 退到 `start`（多字节字符内部没有边界），回调空切片后
                    // `start = end` 永不前进 ⟹ 死循环。退到 start 时改为前进到
                    // **下一个**字符边界，保证每轮至少切出一个完整字符。
                    while end > start && !chunk.is_char_boundary(end) {
                        end -= 1;
                    }
                    if end == start {
                        let mut next = start + 1;
                        while next < chunk.len() && !chunk.is_char_boundary(next) {
                            next += 1;
                        }
                        end = next;
                    }
                }
                f(&bytes[start..end], m_start + start);
                start = end;
            }
        }

        // 尾部间隙：最后一个匹配之后剩下的部分
        if covered_to < line.len() {
            return Err(PreTokenizeError::gap(&line[covered_to..], covered_to));
        }
        Ok(())
    }

    /// 预分词（**不**返回正则错误）。
    ///
    /// A2：本方法为保持 `encode` 系列的公开签名（`-> Encoding` / `-> Vec<i32>`，
    /// 历史上刻意不返回 `Result`）而保留便利形式。遇到正则运行期错误时
    /// **会通过 `eprintln!` + `tracing::error!` 大声报警**，绝不静默 ——
    /// 与 `reader.rs` 的 M4 修法保持同一风格（默认无 tracing subscriber，
    /// 故 `eprintln!` 是保证可见的那条通道）。
    ///
    /// 需要把错误当作硬失败处理（训练路径）的调用方请用
    /// [`Self::tokenize_with_checked`]。
    #[inline(always)]
    pub fn tokenize_with<'a, F>(&self, line: &'a str, f: F)
    where
        F: FnMut(&'a [u8], usize),
    {
        if let Err(e) = self.tokenize_with_checked(line, f) {
            report_pretokenize_error(&e);
        }
    }

    /// 预分词并收集为 Vec。错误处理同 [`Self::tokenize_with`]（大声报警）。
    // pre_tokenize 也顺便改一下以防编译报错（如果训练阶段用到了它）
    #[inline(always)]
    pub fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<(&'a [u8], usize)> {
        let mut chunks = Vec::new();
        self.tokenize_with(text, |chunk, offset| chunks.push((chunk, offset)));
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf8_safe_chunking() {
        let pt = PreTokenizer::new(DEFAULT_PATTERN).unwrap();

        // 构造一个超长纯中文文本（完全没有空格和标点），强制触发 512 字节切断防御
        // "我" 的 UTF-8 编码是 3 个字节
        let long_text = "我".repeat(300); // 900 字节

        let mut chunks = Vec::new();
        pt.tokenize_with(&long_text, |chunk, offset| {
            chunks.push((chunk.to_vec(), offset));
        });

        // 验证 1：有没有被切成多块？(900 字节应该被切成两块：510 + 390)
        assert!(chunks.len() >= 2, "超长文本必须被截断防御机制切分开！");

        // 验证 2：【最核心】每一块必须都是完全合法的 UTF-8 字符串！不能有被劈成两半的汉字
        for (chunk_bytes, offset) in chunks {
            let s = std::str::from_utf8(&chunk_bytes);
            assert!(
                s.is_ok(),
                "灾难！UTF-8 边界切分错误，产生了非法的破损字节！偏移量: {}",
                offset
            );
        }
    }

    // R4-2：`max_chunk_bytes=1/2` 对多字节字符曾**死循环**（end 退到 start 后
    // 永不前进）。本测试若挂住即回归；正常应在毫秒级完成并产出合法切分。
    #[test]
    fn r4_2_tiny_max_chunk_bytes_loops_not_and_splits_on_char_boundary() {
        for cap in [1usize, 2] {
            let pt = PreTokenizer::with_max_chunk_bytes(DEFAULT_PATTERN, cap).unwrap();
            let text = "中a文b"; // 3 字节 + 1 字节交替
            let mut chunks: Vec<Vec<u8>> = Vec::new();
            pt.tokenize_with(text, |bytes, _off| chunks.push(bytes.to_vec()));
            assert!(!chunks.is_empty(), "cap={cap} 必须产出 chunk（死循环回归）");
            // 每个 chunk 必须是合法 UTF-8 且非空（空切片回调正是死循环的成因）
            for c in &chunks {
                assert!(!c.is_empty(), "cap={cap} 不允许产出空 chunk");
                assert!(
                    std::str::from_utf8(c).is_ok(),
                    "cap={cap} 切分必须落在字符边界"
                );
            }
            // 拼接必须还原原文
            let joined: Vec<u8> = chunks.concat();
            assert_eq!(joined, text.as_bytes(), "cap={cap} 切分后必须无损");
        }
    }

    // R4-3：未覆盖间隙的错误消息截断点曾可能落在多字节字符内部 ⟹ panic。
    // 构造一个 >60 字节且第 60 字节在汉字内部的 gap（pattern 只匹配 ASCII 字母）。
    #[test]
    fn r4_3_long_multibyte_gap_errors_not_panics() {
        let pt = PreTokenizer::new(r"[a-z]+").unwrap();
        // 24 个汉字 = 72 字节 > SAMPLE(60)，且第 60 字节落在某个汉字内部
        let text = "中".repeat(24) + "abc";
        let err = pt
            .tokenize_with_checked(&text, |_, _| {})
            .expect_err("未覆盖间隙必须返回错误");
        let msg = err.to_string();
        assert!(msg.contains("未覆盖"), "错误消息应说明未覆盖: {msg}");
        // 消息里必须带样本且不 panic 构造成功
        assert!(msg.contains('中'), "样本应含缺失内容: {msg}");
    }

    #[test]
    fn r4_3_floor_char_boundary_keeps_multibyte_intact() {
        let s = "中文abc";
        // "中" 占 3 字节：下标 1/2 都不是边界，应取到 0
        assert_eq!(floor_char_boundary(s, 1), 0);
        assert_eq!(floor_char_boundary(s, 2), 0);
        assert_eq!(floor_char_boundary(s, 3), 3);
        assert_eq!(floor_char_boundary(s, 99), s.len());
    }
}

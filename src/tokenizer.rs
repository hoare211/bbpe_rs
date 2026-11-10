use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use tracing::info; // 新增

use crate::error::BbpeError;
use crate::pre_tokenizer::PreTokenizer;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

enum ChunkJob<'a> {
    Special {
        id: i32,
        token_str: &'a str,
        offset: (usize, usize),
    },
    Normal {
        bytes: &'a [u8],
        offset: usize,
    },
}

// ---------- 新增：状态机属性 ----------
#[derive(Clone, Copy)]
pub struct FSMProperties {
    pub provides: usize, // 该 Token 开头提供了多少个延续字节 (10xxxxxx)
    pub hangs: usize,    // 该 Token 结尾悬空了多少个字节未补齐
}

// 辅助函数：判断该 Token 是否符合当前的 pending 状态
#[inline(always)]
fn is_valid_for_pending(provides: usize, len: usize, pending: usize) -> bool {
    if pending == 0 {
        return provides == 0; // 状态健康时，绝不接受无头延续字节
    }
    if provides > 0 && provides <= pending {
        if provides == pending {
            return true; // 完美满足了前序悬空，后续无论是 ASCII 还是新 Starter 都合法
        } else {
            return provides == len; // 如果没满足完，那这个 Token 必须纯粹由延续字节组成！
        }
    }
    false
}

/// UTF-8 前导字节 → 还需多少个延续字节（R4-17）。
///
/// **保证边界（第五轮收敛）**：本函数只校验*前导字节区间*
/// （C2..=DF / E0..=EF / F0..=F4），**不是**序列级完整 UTF-8 校验器：
/// surrogate（`ED A0 80`）与超范围标量（`F4 90 80 80`）在计数层面仍会被
/// “当作完整字符消费”。FSM 的承诺是“悬空前导必须被补齐、非法前导不开启
/// 悬空”，而非杜绝一切非法序列（后者取决于词表内容，以 decode 为准）。
/// 所有声明处（Python docstring / 使用说明）均按此口径措辞；类名
/// `UTF8StrictLogitsProcessor` 为历史名称，保留以兼容 API。
///
/// 返回 0 表示**不是**合法前导字节。原实现把 `192..=223` 与 `240..=247`
/// 整段当作合法前导，放行了 C0/C1（overlong 编码）与 F5–F7（超出 U+10FFFF，
/// RFC 3629 明令保留）—— 于是 `C0 80` 这类非法序列会被 FSM 当作"合法
/// 2 字节字符"放行。
#[inline(always)]
fn utf8_lead_hangs(b: u8) -> usize {
    match b {
        0xC2..=0xDF => 1, // 2 字节字符（排除 overlong 的 C0/C1）
        0xE0..=0xEF => 2, // 3 字节字符
        0xF0..=0xF4 => 3, // 4 字节字符（排除超范围的 F5–F7）
        _ => 0,           // ASCII / 延续字节 / 非法字节：不开启悬空
    }
}

/// 由 token 字节计算 FSM 属性（provides / hangs）。**唯一实现** ——
/// `BBPETokenizer::load_with_options` 与本文件的单测共用，杜绝两份逻辑漂移
/// （R4-17 的字节区间修正曾因测试里另抄一份而只修了一半）。
#[inline(always)]
fn fsm_props_of(bytes: &[u8]) -> FSMProperties {
    let len = bytes.len();
    // provides = 开头连续延续字节（10xxxxxx）个数
    let mut provides = 0;
    while provides < len && (128..=191).contains(&bytes[provides]) {
        provides += 1;
    }
    // hangs = 结尾悬空（未补齐）的字符还需多少延续字节
    let mut hangs = 0;
    for &b in &bytes[provides..] {
        if hangs > 0 {
            if (128..=191).contains(&b) {
                hangs -= 1; // 满足一个
            } else {
                hangs = utf8_lead_hangs(b); // 坏死重置，按当前字节重新判定
            }
        } else {
            hangs = utf8_lead_hangs(b);
        }
    }
    FSMProperties { provides, hangs }
}

/// R4-5：模型 id 跨度预算。`BBPETokenizer` 按 `max_id + 1` 分配两份稠密数组
/// （`fsm_props_array` 16 B + `vocab_strings` 24 B ≈ 40 B/条），合法模型的
/// id 跨度与词表规模相称（256 + merges + specials）。含 `"2147483647": [0]`
/// 这类**结构合法但稀疏**的模型会在加载期尝试 GB 级分配（不可信模型目录的
/// OOM DoS），且 `max_id + 1` 在 `i32::MAX` 时溢出（debug panic / release 回绕）。
/// 这里用 i64 预算统一拒绝，加载路径（validate=false 亦生效）提前失败。
pub(crate) const MAX_MODEL_ID_SPAN: i64 = 1 << 24; // 16,777,216 条 ⟹ 稠密数组 ~670 MB 上限

/// 拒绝"稀疏大 id"模型（R4-5）。`BpeState::load_with_options` 与
/// `BBPETokenizer::load_with_options`（程序内构造的 state）共用。
pub(crate) fn reject_sparse_ids(vocab: &HashMap<i32, Vec<u8>>) -> Result<(), BbpeError> {
    let max_id = vocab.keys().copied().max().unwrap_or(0) as i64;
    if max_id + 1 > MAX_MODEL_ID_SPAN {
        return Err(BbpeError::InvalidModelFormat(format!(
            "vocab 最大 id {max_id} 超过加载预算（最大 {}）：该值会触发约 {} MB 的稠密数组分配，\
             疑似稀疏大 id 的损坏或恶意模型（合法模型的 id 跨度与词表规模相称）",
            MAX_MODEL_ID_SPAN - 1,
            MAX_MODEL_ID_SPAN * 40 / 1_000_000
        )));
    }
    Ok(())
}

// 你的原有结构体不变
/// 解析模型 JSON 中的 id 字符串。**拒绝负数**（M1）。
///
/// 负数在 `as usize` 转换后会变成 `usize::MAX` 量级的巨值，进而在
/// `BBPETokenizer` 的 O(1) 预计算数组处造成越界 panic。契约要求这类损坏
/// 必须在解析期被拒绝，而不是让 panic 发生时才暴露。
fn parse_nonneg_id(s: &str) -> Option<i32> {
    match s.parse::<i32>().ok() {
        Some(v) if v >= 0 => Some(v),
        _ => None,
    }
}

#[derive(Eq, PartialEq)]
struct MergeItem {
    rank: i32,
    pos: usize,
    id0: i32,
    id1: i32,
}

impl Ord for MergeItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for MergeItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------- 编码热路径优化（2026-11-05）----------
//
// 实测定位：10MB 中文里 encode 有 92% 时间在 BPE 合并，成本可拟合为
//   `0.174 µs/字节 + 0.214 µs/chunk`；chunk 平均只有 8.9 字节 ⟹ 每个 chunk 的
// 固定开销不可忽略。原实现每个 chunk 都要新建 5 个结构（ids / prev / next /
// deleted / BinaryHeap），且每个相邻 pair 都要对 `HashMap<(i32,i32),i32>` 做一次
// SipHash 查找。下面两处改动分别消除这两项：
//   1. 线程局部工作区复用（不再 per-chunk 分配）
//   2. 打包成 u64 的 FxHashMap 索引（一次哈希替代元组哈希 + SipHash）

/// 把 (p0, p1) 打包成 u64：高 32 位 p0、低 32 位 p1。id 恒为非负，直接转 u32。
#[inline(always)]
fn pack_pair_key(p0: i32, p1: i32) -> u64 {
    ((p0 as u32 as u64) << 32) | (p1 as u32 as u64)
}

/// 每线程复用的 BPE 工作区。`clear()` 保留容量，热路径上零分配。
#[derive(Default)]
struct BpeScratch {
    ids: Vec<i32>,
    prev: Vec<isize>,
    next: Vec<isize>,
    deleted: Vec<bool>,
    offsets: Vec<(usize, usize)>,
    heap: BinaryHeap<MergeItem>,
}

thread_local! {
    static BPE_SCRATCH: RefCell<BpeScratch> = RefCell::new(BpeScratch::default());
}

// ---------- chunk 级 memo（2026-11-05）----------
//
// 实测 chunk 重复度：10MB 中文里 1,117,445 个 chunk 只有 248,242 个唯一值
// （理想命中率上限 77.8%，top10 个 chunk 占 33.7%）⟹ 短 chunk 的合并结果
// 高度可复用。纯汉字长段（510B chunk）命中率为 0，故只缓存短 chunk。
//
// 只在**裸 ID 路径**（`track == None`）生效：带 offsets 的调用需要逐 token 的
// 字节区间，无法直接复用缓存。缓存按 tokenizer 实例隔离（`owner`），避免
// 同一线程先后使用两个模型时串味。
const MEMO_MAX_CHUNK: usize = 32; // 只缓存 ≤32 字节的 chunk
const MEMO_CAP: usize = 200_000; // 每线程最多缓存条目数（约 30~40 MB 上限）

#[derive(Default)]
struct ChunkMemo {
    owner: u64,
    map: FxHashMap<Box<[u8]>, Box<[i32]>>,
}

thread_local! {
    static CHUNK_MEMO: RefCell<ChunkMemo> = RefCell::new(ChunkMemo::default());
}

#[derive(Debug, Clone)]
pub struct Encoding {
    pub ids: Vec<i32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
}

// A8：`TokenType` / `SftEncoding` 已随 `encode_with_types` 一并删除
// （零调用者，且其"按 token 类型掩码"语义与 Python 侧实际使用的
//   "按对话轮次角色掩码"不同 —— 后者才是 SFT 的正确语义）。
// 详见 `BBPETokenizer::encode_ordinary_batch` 上方的说明。

// =========================================================
// 模型契约 v1（Model Contract v1，见 docs/v2输入输出标准规范技术方案.md §7）
// =========================================================

/// 模型格式标识。`ExtendedTokenizerModel::format` 的固定值（MUST）。
pub const MODEL_FORMAT: &str = "bbpe-rs-model";

/// 当前模型契约版本（MUST）。加载器 **MUST** 拒绝更高的 `format_version`。
pub const FORMAT_VERSION: u32 = 1;

/// 编译期注入的 git commit（由 `build.rs` 提供；取不到时为 `unknown`）。
pub fn git_commit() -> &'static str {
    option_env!("GIT_COMMIT").unwrap_or("unknown")
}

/// 语料侧统计（§7.1 `meta.corpus`，配合 §6.6 计数上报）
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusMeta {
    /// reader 类型：`text` / `jsonl:text` / `jsonl:conversations` / `jsonl:dynamic`
    pub reader: String,
    /// 物理行数（含空行与坏行）
    pub total_lines: u64,
    /// 读入的原始字节数（剥离行尾前）
    pub total_bytes: u64,
    /// 被静默跳过的行数（逐 reader 语义见规范 §6.3）
    pub skipped_lines: u64,
    /// 非法 UTF-8 被替换为 U+FFFD 的次数（§6.2）
    pub replacement_chars: u64,
    /// R4-7：读取中途 IO 错误次数。`> 0` 表示语料被**截断**（其后内容未读），
    /// 与"正常读到 EOF"有本质区别。旧模型无此字段 ⟹ 反序列化时取 0。
    #[serde(default)]
    pub io_errors: u64,
}

/// 采样参数（§7.1 `meta.sampling`）。仅采样训练（v7 剖面）写出；全量训练时缺省。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingMeta {
    pub rate: f64,
    pub seed: u64,
    pub heldout_rate: f64,
    pub heldout_seed: u64,
}

/// 训练参数与结果（§7.1 `meta.train`）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainMeta {
    pub vocab_size: usize,
    pub min_frequency: u32,
    /// 实际完成的 merges 条数
    pub actual_merges: usize,
    /// `vocab_size_reached` | `min_frequency_terminated` | `no_more_pairs`
    ///
    /// `no_more_pairs`（M2 新增）：候选堆与 global_counts 双双耗尽 ⟹ 真的没有
    /// 可合并的 pair 了。此前该情形被 `best_pair_u64 == 0` 哨兵误归并进
    /// `min_frequency_terminated`，导致 NUL 语料下终止原因失真。
    pub end_reason: String,
    pub max_chunk_bytes: usize,
    /// m6：本模型若由更大模型**截断**而来，记录源模型的 `vocab_size`。
    /// `None` = 直接训练产物。
    ///
    /// 存在意义：截断后 `vocab_size` 仍是**训练时**的目标值（事实，不应改写），
    /// 但只看 `vocab_size` 会让人误以为模型有那么大。有了 `truncated_from`
    /// 就能一眼看出"这是从 X 截断到 Y 的"，且 `actual_merges` 才是真实规模。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncated_from: Option<usize>,
}

/// 评估结果（§7.1 `meta.eval`，MAY）
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub held_out_bytes: Option<u64>,
    // M3：同 DTO 主字段，改用 BTreeMap 保证 meta 块亦可逐字节比对
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fertility_at_k: Option<BTreeMap<String, f64>>,
}

/// 模型元信息（§7.1）。**纯记录**：加载器 MUST 忽略未知字段（ADR-4），
/// 缺失任何子块均不影响加载。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbpe_rs_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    /// 产出该模型的产品剖面：`v2` | `v7`（§9）。**仅为元信息**，不代表格式差异（ADR-1）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corpus: Option<CorpusMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train: Option<TrainMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval: Option<EvalMeta>,
}

// 1. 统一的数据传输对象 (DTO)，同时支持序列化与反序列化
#[derive(Serialize, Deserialize)]
pub struct ExtendedTokenizerModel {
    /// MUST（v1 起）。缺失 ⟹ v0 旧格式（迁移模式，见规范 §7.5）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// MUST（v1 起）。缺失 ⟹ v0；**未知且 > 1** ⟹ `UnsupportedFormatVersion`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format_version: Option<u32>,
    // M3：序列化容器由 HashMap 改为 BTreeMap —— std HashMap 的迭代序受 RandomState
    // 进程级随机种子影响，会导致"同配置重训"产出的模型文件字节序不同，
    // 使 `SaveOptions::no_meta` 承诺的逐字节比对失效。BTreeMap 按键字典序输出，
    // 跨进程/跨运行稳定。反方向（加载）同样受益：读入顺序确定。
    pub vocab: BTreeMap<String, Vec<u8>>,
    pub merges: BTreeMap<String, i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub added_tokens: Option<BTreeMap<String, i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_tokenizer_pattern: Option<String>,
    /// MAY。纯记录块，未知字段 MUST 被忽略
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<ModelMeta>,
}

/// 保存选项（§7.6）
#[derive(Debug, Clone, Default)]
pub struct SaveOptions {
    /// 完全省略 `meta` 块：使"同配置重训"的模型文件可逐字节比对（配合确定性测试 T-7）
    pub no_meta: bool,
}

/// 加载选项（§7.4 兼容矩阵）
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// 豁免 v0 模型的 `pre_tokenizer_pattern` 缺失（回退 DEFAULT_PATTERN 并打印警告）。
    /// **仅豁免 Q2**，V1–V5 自洽性校验照常执行（规范 §7.4）。
    pub allow_legacy: bool,
    /// 加载时执行 V1–V5 自洽性校验。默认 `true`。
    pub validate: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            allow_legacy: false,
            validate: true,
        }
    }
}

// 2. 统一的数据载体 (Single Source of Truth)
#[derive(Clone)]
pub struct BpeState {
    pub vocab: HashMap<i32, Vec<u8>>,
    pub merges: HashMap<(i32, i32), i32>,
    /// `merges` 的打包快速索引（`pack_pair_key` 生成的 u64 → new_id），仅供编码热路径。
    /// 与 `merges` 必须同步：在 [`BpeState::load`] / [`BpeState::truncate_to_k`] 以及
    /// trainer 构造 state 之后调用 [`BpeState::rebuild_merges_fast`]。
    pub(crate) merges_fast: FxHashMap<u64, i32>,
    pub special_tokens: HashMap<String, i32>,
    pub pre_tokenizer_pattern: String,
    /// 模型元信息（MAY）。保存时写出，加载时原样带回；不参与任何算法逻辑。
    pub meta: Option<ModelMeta>,
    /// 该文件实际的 `format_version`（加载时记录；`None` = 新建或 v0 旧格式）。
    /// 保存时一律写 [`FORMAT_VERSION`]，与本字段无关。
    pub format_version: Option<u32>,
    /// A4：该文件实际声明的 `format`（加载时记录；`None` = v0 旧格式未声明）。
    ///
    /// 此前**只**记录 `format_version`，导致 `inspect` 拿不到真实值，只能把
    /// 常量 [`MODEL_FORMAT`] 打印出来 —— 于是无论文件里有没有 `format`，输出
    /// 都是 `bbpe-rs-model`。契约诊断工具在契约字段上误报，比不诊断更糟。
    pub format: Option<String>,
    /// m8：加载时是否发生过 `pre_tokenizer_pattern` 缺失回退（Q2 静默 fallback）。
    ///
    /// 纯运行时标记，不参与序列化（`BpeState` 本身不实现 Serialize）。
    /// 存在意义：回退时 Rust 侧只发 `tracing::warn`，而 Python 默认未装
    /// subscriber ⟹ 用户完全看不到"正在用错误的 pattern 分词"。绑定层据此
    /// 发出 Python 可见的告警。
    pub legacy_pattern_fallback: bool,
}

impl BpeState {
    /// 重建编码热路径用的打包索引。任何改动过 `merges` 的操作都必须调用一次。
    pub(crate) fn rebuild_merges_fast(&mut self) {
        let mut fast = FxHashMap::default();
        fast.reserve(self.merges.len());
        for (&(p0, p1), &new_id) in &self.merges {
            fast.insert(pack_pair_key(p0, p1), new_id);
        }
        self.merges_fast = fast;
    }

    /// 热路径查表：一次 u64 哈希（FxHash）替代 `(i32,i32)` 元组的 SipHash。
    #[inline(always)]
    fn merge_lookup(&self, p0: i32, p1: i32) -> Option<i32> {
        self.merges_fast.get(&pack_pair_key(p0, p1)).copied()
    }

    pub fn load(filepath: &str) -> Result<Self, BbpeError> {
        Self::load_with_options(filepath, LoadOptions::default())
    }

    /// 按加载选项读取模型（§7.4 兼容矩阵）。
    ///
    /// - `format_version` 缺失 ⟹ v0 旧格式；**未知且 > [`FORMAT_VERSION`]** ⟹ `UnsupportedFormatVersion`
    /// - `pre_tokenizer_pattern` 缺失 ⟹ 默认报错 `MissingPreTokenizerPattern`；
    ///   `allow_legacy = true` 时回退 DEFAULT_PATTERN 并打印警告（**仅豁免此条**）
    /// - `validate = true` 时执行 V1–V5 自洽性校验
    pub fn load_with_options(filepath: &str, opts: LoadOptions) -> Result<Self, BbpeError> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let model: ExtendedTokenizerModel = serde_json::from_reader(reader)?;

        // ---- format / format_version（§7.2）----
        // v1 起两者 MUST 同时存在：半新半旧的组合（只有其一）更可能是损坏或
        // 手工编辑的文件，直接报 `InvalidModelFormat` 而非按 v0 宽容处理
        match (&model.format, model.format_version) {
            (Some(f), Some(v)) => {
                if f != MODEL_FORMAT {
                    return Err(BbpeError::InvalidModelFormat(format!(
                        "未知 format: {f:?}（期望 {MODEL_FORMAT:?}）"
                    )));
                }
                if v > FORMAT_VERSION {
                    return Err(BbpeError::UnsupportedFormatVersion(v, FORMAT_VERSION));
                }
            }
            (Some(f), None) => {
                return Err(BbpeError::InvalidModelFormat(format!(
                    "模型含 format={f:?} 但缺 format_version（v1 起两者必须同时存在）"
                )));
            }
            (None, Some(v)) => {
                return Err(BbpeError::InvalidModelFormat(format!(
                    "模型含 format_version={v} 但缺 format（v1 起两者必须同时存在）"
                )));
            }
            (None, None) => info!("加载 v0 旧格式模型（无 format/format_version）: {filepath}"),
        }
        let meta = model.meta;

        let mut merges = HashMap::new();
        for (k, v) in model.merges {
            // 坏数据必须报错而非静默吞掉（unwrap_or(0) 会把损坏模型伪装成合法空表，
            // 污染 train_vocab 的推论 7 判据等一切回读逻辑）
            let parts: Vec<&str> = k.split(',').collect();
            if parts.len() != 2 {
                return Err(BbpeError::InvalidModelFormat(format!(
                    "merges key 应为 \"p0,p1\": {k:?}"
                )));
            }
            let p0 = parse_nonneg_id(parts[0]).ok_or_else(|| {
                BbpeError::InvalidModelFormat(format!("merges key 的 p0 不是非负整数: {k:?}"))
            })?;
            let p1 = parse_nonneg_id(parts[1]).ok_or_else(|| {
                BbpeError::InvalidModelFormat(format!("merges key 的 p1 不是非负整数: {k:?}"))
            })?;
            merges.insert((p0, p1), v);
        }

        let mut vocab = HashMap::new();
        for (k, v) in model.vocab {
            // M1：负 id 必须在此处拒绝。`id as usize` 对负数得到 usize::MAX 量级的巨值，
            // 会在 BBPETokenizer 的 O(1) 数组构建处越界 panic。V1–V5 只检查"存在性"
            // 不检查符号，故此处的解析期拒绝是最后一道防线（validate=false 时亦生效）。
            let id = parse_nonneg_id(&k).ok_or_else(|| {
                BbpeError::InvalidModelFormat(format!("vocab key 不是非负整数 id: {k:?}"))
            })?;
            vocab.insert(id, v);
        }

        // M3：DTO 用 BTreeMap 保证序列化键序确定；BpeState 内部仍用 HashMap（查找更快）
        let special_tokens: HashMap<String, i32> =
            model.added_tokens.unwrap_or_default().into_iter().collect();
        // C1（2026-10-12 评审）：空 special token（""）无任何语义（零字节、解码为空），
        // 且 AhoCorasick 的空 pattern 在每个字节位都匹配（含 UTF-8 字符内部的延续
        // 字节位置）⟹ encode/encode_parallel 的 `&text[start..end]` 在非字符边界
        // 切片直接 panic。V1–V5 均不涉及空串，故这是解析期防线（validate=false
        // 时同样生效），与负 id 的 parse_nonneg_id 拒绝同款。
        for (text, &id) in &special_tokens {
            if text.is_empty() {
                return Err(BbpeError::InvalidModelFormat(format!(
                    "added_tokens 含空字符串 special token（id {id}）：空 pattern 会在每个字节位匹配，\
                     encode 必崩；空 token 无保留价值，请删除该条目"
                )));
            }
        }
        for (text, &id) in &special_tokens {
            vocab.insert(id, text.as_bytes().to_vec());
        }

        // R4-5：稀疏大 id 提前拒绝（在分配任何稠密数组之前）
        reject_sparse_ids(&vocab)?;

        // ---- pre_tokenizer_pattern：缺失是静默错误源（Q2），v1 起默认报错 ----
        // m8：同时置位 `legacy_pattern_fallback`，让调用方（尤其 Python 绑定，
        // 默认无 tracing subscriber）能把"正在用错误 pattern"这件事暴露给用户。
        let mut legacy_pattern_fallback = false;
        let pre_tokenizer_pattern = match model.pre_tokenizer_pattern {
            Some(p) => p,
            None if opts.allow_legacy => {
                legacy_pattern_fallback = true;
                tracing::warn!(
                    "模型 {filepath} 缺少 pre_tokenizer_pattern（v0 旧格式），已回退 DEFAULT_PATTERN。\
                     若该模型训练时使用了自定义 pattern，加载结果将是错的 —— 请用 \
                     `inspect_model --migrate <in> <out> --pattern <regex>` 迁移。"
                );
                crate::pre_tokenizer::DEFAULT_PATTERN.to_string()
            }
            None => return Err(BbpeError::MissingPreTokenizerPattern),
        };

        let mut state = Self {
            vocab,
            merges,
            // 下面立刻 rebuild，先占位避免半初始化状态被误用
            merges_fast: FxHashMap::default(),
            special_tokens,
            pre_tokenizer_pattern,
            meta,
            format: model.format,
            format_version: model.format_version,
            // m8：记录本次加载是否走了 DEFAULT_PATTERN 回退，供绑定层告警
            legacy_pattern_fallback,
        };
        state.rebuild_merges_fast();
        // 默认开启自洽性校验（ADR-2）；allow_legacy 仅豁免 pattern 缺失，不豁免 V1–V5（§7.4）
        if opts.validate {
            state.validate()?;
        }
        Ok(state)
    }

    /// 按 BPE 贪心前缀性质截断：保留前 k 次合并（`new_id < 256 + k`）。
    ///
    /// 依据（方案 §19.1，实测确证 100,000/100,000 逐位一致）：
    ///   1. `new_id = 256 + i` 把合并次序编码进 id ⟹ 按 id 截断即得前 k 次合并的词表；
    ///   2. 第 i 次 merge 的输入 token 只可能是基础字节或前 i-1 次合并的产物，
    ///      即 key `(a,b)` 必满足 `a,b < new_id < limit` ⟹ retain 后词表自洽闭合。
    ///
    /// 返回实际保留的 merges 数 `k' = min(k, self.merges.len())`（超出 K_max 自动钳制）。
    ///
    /// special_tokens 位于词表尾部（id ≥ base_vocab_size），截断会把它们从 vocab 中
    /// 删除；本方法会**重新注入**（与 `BpeState::load` 的行为一致），保证截断后的
    /// state 自身仍满足 V5 —— `validate()` 对 `truncate_to_k` 的产物保持幂等。
    pub fn truncate_to_k(&mut self, k: usize) -> usize {
        let k = k.min(self.merges.len());
        let limit = (256 + k) as i32;
        self.vocab.retain(|&id, _| id < limit);
        self.merges.retain(|_, &mut nid| nid < limit);
        // merges 变了 ⟹ 打包索引必须同步，否则热路径仍按旧词表合并
        self.rebuild_merges_fast();
        for (text, &id) in &self.special_tokens {
            if id >= limit {
                self.vocab.insert(id, text.as_bytes().to_vec());
            }
        }
        k
    }

    /// 模型自洽性校验 **V1–V5**（规范 §7.3）。任一条不满足 ⟹ [`BbpeError::ModelValidation`]。
    ///
    /// 各条均**可从模型自身推导**，不依赖训练时参数（`base_vocab_size` 等在截断后不可恢复，故不采用）：
    /// - V1 vocab 含 0–255 全部基础字节，**且不含负 id**
    /// - V2 每条 merge 的 `(p0, p1)` 与 `new_id` 均在 vocab 中（词表闭合）
    /// - V3 merges 的 `new_id` 集合恰为连续区间 `{256, …, 256+m−1}`（保证"按 id 截断 = 前 k 次合并"）
    /// - V4 `pre_tokenizer_pattern` 可编译
    /// - V5 `added_tokens` id 互不重复、均大于最大 `new_id`（尾部性质），且字节与 vocab 中同 id 条目一致
    /// - V6 `added_tokens` 文本非空（空 special token 会使 encode panic，C1）
    /// - V7 每条 merge 的产物字节 == `vocab[p0] + vocab[p1]`（R4-6，否则 round-trip 被破坏）
    pub fn validate(&self) -> Result<(), BbpeError> {
        // ---- V1：基础字节完整性 + 非负 id（M1）----
        let mut missing: Vec<i32> = (0..256).filter(|id| !self.vocab.contains_key(id)).collect();
        if !missing.is_empty() {
            missing.truncate(8);
            return Err(BbpeError::ModelValidation(format!(
                "V1 违反：vocab 缺少基础字节 {missing:?}（字节级 BPE 必须含 0-255 全部字节）"
            )));
        }
        // M1：负 id 在 `id as usize` 后会变成巨值下标导致 panic。加载路径已在解析期
        // 拒绝，此处是第二道防线 —— 覆盖程序内构造/修改后的 state（如手工 insert）。
        if let Some(&neg) = self.vocab.keys().find(|&&id| id < 0) {
            return Err(BbpeError::ModelValidation(format!(
                "V1 违反：vocab 含负 id {neg}（id 必须 ≥ 0，否则 `id as usize` 会越界 panic）"
            )));
        }
        if let Some((&(p0, p1), _)) = self.merges.iter().find(|((a, b), _)| *a < 0 || *b < 0) {
            return Err(BbpeError::ModelValidation(format!(
                "V1 违反：merge 的 key ({p0},{p1}) 含负 id（id 必须 ≥ 0）"
            )));
        }

        // ---- V2：词表闭合性 ----
        for (&(p0, p1), &new_id) in &self.merges {
            for id in [p0, p1, new_id] {
                if !self.vocab.contains_key(&id) {
                    return Err(BbpeError::ModelValidation(format!(
                        "V2 违反：merge ({p0},{p1})->{new_id} 引用的 id {id} 不在 vocab 中（词表不闭合）"
                    )));
                }
            }
        }

        // ---- V3：new_id 连续性（同时覆盖"重复 new_id"：重复必导致某位不等于期望值）----
        let mut new_ids: Vec<i32> = self.merges.values().copied().collect();
        new_ids.sort_unstable();
        for (i, &nid) in new_ids.iter().enumerate() {
            let expected = 256 + i as i32;
            if nid != expected {
                return Err(BbpeError::ModelValidation(format!(
                    "V3 违反：merges 的 new_id 集合不是连续区间 {{256, …, {}}}（第 {} 个为 {nid}，期望 {expected}）",
                    256 + new_ids.len() as i32 - 1,
                    i
                )));
            }
        }

        // ---- V4：pattern 可编译 ----
        crate::pre_tokenizer::PreTokenizer::new(&self.pre_tokenizer_pattern).map_err(|e| {
            BbpeError::ModelValidation(format!("V4 违反：pre_tokenizer_pattern 无法编译: {e}"))
        })?;

        // ---- V7：merge 产物字节 == 两个输入 token 字节之和（R4-6）----
        // V1/V2 只检查 id 存在与引用闭合，**不检查字节**。于是 `merges` 含
        // `"97,98": 256` 而 `vocab["256"]` 错写为 `[120]` 时，validate 全部
        // 通过，但 `"ab"` 编码为 256 后解码成 `"x"`（round-trip 被破坏且
        // inspect --validate-only 看不出任何问题）。训练器按
        // `vocab[new_id] == vocab[p0] + vocab[p1]` 构造（trainer.rs），
        // 故合法模型必须满足该恒等；截断（truncate_to_k）只做保留，同样满足。
        for (&(p0, p1), &new_id) in &self.merges {
            let mut expect: Vec<u8> = Vec::new();
            for id in [p0, p1] {
                // V2 已保证存在；此处再兜底一次，避免直接索引 panic
                match self.vocab.get(&id) {
                    Some(b) => expect.extend_from_slice(b),
                    None => continue,
                }
            }
            match self.vocab.get(&new_id) {
                Some(actual) if *actual == expect => {}
                _ => {
                    return Err(BbpeError::ModelValidation(format!(
                        "V7 违反：merge ({p0},{p1})->{new_id} 的产物字节与输入 token 字节之和不一致（\
                         期望 {:?}，实际 {:?}）—— 编码/解码 round-trip 会被破坏",
                        expect,
                        self.vocab.get(&new_id)
                    )));
                }
            }
        }

        // ---- V5：special tokens 与 vocab / merges 三方一致（Q7 落地）----
        let max_new_id = new_ids.last().copied().unwrap_or(255);
        let mut seen: HashMap<i32, &String> = HashMap::with_capacity(self.special_tokens.len());
        for (text, &id) in &self.special_tokens {
            if id <= max_new_id {
                return Err(BbpeError::ModelValidation(format!(
                    "V5 违反：special token {text:?} 的 id {id} 不大于最大 merge new_id {max_new_id}（应在尾部）"
                )));
            }
            if let Some(prev) = seen.insert(id, text) {
                return Err(BbpeError::ModelValidation(format!(
                    "V5 违反：special token {prev:?} 与 {text:?} 的 id 均为 {id}（id 必须互不重复）"
                )));
            }
            match self.vocab.get(&id) {
                Some(bytes) if bytes == text.as_bytes() => {}
                Some(bytes) => {
                    return Err(BbpeError::ModelValidation(format!(
                        "V5 违反：special token {text:?} 的 id {id} 在 vocab 中的字节为 {bytes:?}，与其文本字节不一致"
                    )));
                }
                None => {
                    return Err(BbpeError::ModelValidation(format!(
                        "V5 违反：special token {text:?} 的 id {id} 不在 vocab 中"
                    )));
                }
            }
        }

        // ---- V6：special token 文本非空（C1，2026-10-12 评审）----
        // 加载路径已在解析期拒绝空串；此处覆盖程序内构造/修改后的 state
        // （如手工 insert），使"契约放行、运行即崩"的形态在 validate 层闭合。
        if let Some((text, _)) = self.special_tokens.iter().find(|(t, _)| t.is_empty()) {
            return Err(BbpeError::ModelValidation(format!(
                "V6 违反：special token 为空字符串（id {}）—— 空 pattern 在每个字节位匹配，\
                 encode / encode_parallel 的 `&text[start..end]` 会在非字符边界 panic",
                self.special_tokens[text]
            )));
        }

        Ok(())
    }

    pub fn save(&self, filepath: &str) -> Result<(), BbpeError> {
        self.save_with_options(filepath, SaveOptions::default())
    }

    /// 按保存选项写出模型（§7.6）。
    ///
    /// 一律写出 `format = "bbpe-rs-model"` 与 `format_version = 1`（ADR-2）；
    /// `no_meta = true` 时省略整个 `meta` 块，使同配置重训的模型可逐字节比对。
    pub fn save_with_options(&self, filepath: &str, opts: SaveOptions) -> Result<(), BbpeError> {
        // M3：容器用 BTreeMap —— 写出顺序按 key 字典序固定，跨进程/跨运行稳定。
        // 这是 `SaveOptions::no_meta`（"同配置重训可逐字节比对"）得以成立的前提。
        let mut str_vocab = BTreeMap::new();
        for (&id, bytes) in &self.vocab {
            str_vocab.insert(id.to_string(), bytes.clone());
        }

        let mut str_merges = BTreeMap::new();
        for (&(p0, p1), &new_id) in &self.merges {
            str_merges.insert(format!("{},{}", p0, p1), new_id);
        }

        let data = ExtendedTokenizerModel {
            format: Some(MODEL_FORMAT.to_string()),
            format_version: Some(FORMAT_VERSION),
            vocab: str_vocab,
            merges: str_merges,
            added_tokens: if self.special_tokens.is_empty() {
                None
            } else {
                Some(
                    self.special_tokens
                        .iter()
                        .map(|(k, &v)| (k.clone(), v))
                        .collect(),
                )
            },
            pre_tokenizer_pattern: Some(self.pre_tokenizer_pattern.clone()),
            meta: if opts.no_meta {
                None
            } else {
                Some(fill_volatile_meta(self.meta.clone().unwrap_or_default()))
            },
        };

        // R4-9：原子保存。原实现直接 `File::create` **截断**旧模型后流式写出：
        // 序列化中途失败会留下被截断的旧模型（原模型已毁、新模型不完整），
        // 且 `BufWriter` drop 时的 flush 错误**无法上报** ⟹ `save()` 可能对
        // 一个没写完的模型返回 `Ok(())`。改为：同目录临时文件 → flush →
        // sync_all → rename（POSIX rename 原子替换），任何一步失败都保留旧模型。
        let tmp = format!("{filepath}.tmp-{}", std::process::id());
        let write_result = (|| -> Result<(), BbpeError> {
            let file = File::create(&tmp)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, &data)?;
            // flush 必须显式：drop 的 flush 错误会被静默吞掉
            writer.flush()?;
            // 把内核缓冲区落盘，避免"进程退出即丢数据"（掉电 / 容器迁移场景）
            writer.get_ref().sync_all()?;
            Ok(())
        })();
        if let Err(e) = write_result {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        if let Err(e) = std::fs::rename(&tmp, filepath) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
        Ok(())
    }

    /// 取（或创建）`meta` 的可变引用，供调用方记录出身信息（G1 可追溯）。
    pub fn meta_mut(&mut self) -> &mut ModelMeta {
        self.meta.get_or_insert_with(ModelMeta::default)
    }
}

/// 补齐易变字段（`created_at` / `bbpe_rs_version` / `git_commit`），已存在则不覆盖。
fn fill_volatile_meta(mut meta: ModelMeta) -> ModelMeta {
    if meta.created_at.is_none() {
        meta.created_at = Some(rfc3339_now_utc());
    }
    if meta.bbpe_rs_version.is_none() {
        meta.bbpe_rs_version = Some(env!("CARGO_PKG_VERSION").to_string());
    }
    if meta.git_commit.is_none() {
        meta.git_commit = Some(git_commit().to_string());
    }
    meta
}

/// 当前 UTC 时间的 RFC3339 表示（秒精度）。为免引入 chrono 依赖，手写 civil-from-days。
fn rfc3339_now_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let sod = secs % 86_400;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

/// Howard Hinnant 的 civil_from_days：days since 1970-01-01 → (year, month, day)
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `bpe_merge_core` 的 `track` 参数类型：`(token 字符串表, 偏移表)`。
/// 抽出别名以避免 `clippy::type_complexity`，并让签名自解释。
type Track<'a> = (&'a mut Vec<String>, &'a mut Vec<(usize, usize)>);

pub struct BBPETokenizer {
    pre_tokenizer: PreTokenizer,
    pub state: BpeState,

    special_matcher: AhoCorasick,

    // ---------- P1: O(1) 高速寻址预计算缓存表 ----------
    // C3（2026-10-12 评审）：`vocab_bytes` 字段已删除 —— 全仓 grep 确认除构造代码外
    // 零读取者，对 32k–128k 词表是一份全量内存拷贝的纯浪费（与 A8"零调用者即删"同口径）。
    pub vocab_strings: Vec<String>,
    pub fsm_props_array: Vec<FSMProperties>, // 替代原来的 HashMap

    // ---------- 新增：FSM 预计算缓存表 ----------
    pub valid_for_pending_0: Vec<i32>,
    pub valid_for_pending_1: Vec<i32>,
    pub valid_for_pending_2: Vec<i32>,
    pub valid_for_pending_3: Vec<i32>,

    // ---------- 新增：O(1) 特殊字符 ID 查询集 ----------
    pub special_token_ids: HashSet<i32>,

    /// 本实例的唯一标识，用于隔离线程局部 chunk 缓存（防止同线程多模型串味）。
    memo_owner: u64,
}

static NEXT_TOKENIZER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[inline]
fn next_tokenizer_id() -> u64 {
    NEXT_TOKENIZER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// m1：原先是 `unsafe impl Send for BBPETokenizer {}` + `unsafe impl Sync`。
// 该类型的所有字段（PreTokenizer / AhoCorasick / Vec / HashMap / HashSet）本就
// 自动满足 Send + Sync，unsafe impl 纯属冗余；更糟的是它会**掩盖**未来新增的
// 非 Sync 字段（如 Rc / RefCell），把一个编译期错误悄悄变成数据竞争 UB。
// 改用编译期静态断言表达同一意图：既不引入 unsafe，又在字段退化时立即编译失败。
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BBPETokenizer>();
};
// 对外部暴露时，推荐的最佳实践是包装进 Arc
pub type ArcTokenizer = std::sync::Arc<BBPETokenizer>;

impl BBPETokenizer {
    /// 从磁盘加载模型（严格模式：缺 `pre_tokenizer_pattern` 报错，并执行 V1–V5 校验）。
    /// 需要加载 v0 旧模型时用 [`Self::load_with_options`]。
    pub fn load(filepath: &str) -> Result<Self, BbpeError> {
        Self::load_with_options(filepath, LoadOptions::default())
    }

    /// 按加载选项加载模型（模型契约 v1 §7.4）。见 [`BpeState::load_with_options`]。
    pub fn load_with_options(filepath: &str, opts: LoadOptions) -> Result<Self, BbpeError> {
        Self::from_state(BpeState::load_with_options(filepath, opts)?)
    }

    /// 由**内存中的 state** 构建可用分词器（重建 Aho-Corasick 匹配器、FSM 属性表、
    /// 四张合法接续表与预分词器）。
    ///
    /// 为什么单独抽出来：除"从磁盘加载"外，"模型编辑后重建"（[`Self::cut_to_merges`]、
    /// [`Self::with_pattern`]，以及绑定层的同名能力）也需要同一套缓存重建流程。
    /// 若各写一份，两边的 FSM/接续表算法迟早漂移 —— 而这类缓存一旦不同步，
    /// 表现是"编码结果随机不对"，极难定位。
    ///
    /// 入参 state 的合法性由调用方保证（`load_with_options` 已通过 V1–V7 校验；
    /// 本方法只做**不会 panic** 的最后防线检查：负 id、稀疏大 id）。
    pub fn from_state(state: BpeState) -> Result<Self, BbpeError> {
        let patterns: Vec<&String> = state.special_tokens.keys().collect();
        let special_matcher = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .map_err(|e| BbpeError::AhoCorasickError(e.to_string()))?;

        // M1 兜底：validate=false 时负 id 不会被 validate() 拦下，而下面的
        // `id as usize` 会把 -1 变成 usize::MAX 量级下标 ⟹ 直接 panic。
        // 此处显式拒绝，保证任何加载路径都不会 panic。
        if let Some(&neg) = state.vocab.keys().find(|&&id| id < 0) {
            return Err(BbpeError::InvalidModelFormat(format!(
                "vocab 含负 id {neg}（id 必须 ≥ 0）"
            )));
        }
        if let Some((text, &id)) = state.special_tokens.iter().find(|(_, id)| **id < 0) {
            return Err(BbpeError::InvalidModelFormat(format!(
                "special token {text:?} 的 id {id} 为负（id 必须 ≥ 0）"
            )));
        }

        // R4-5：稀疏大 id 预算（覆盖程序内构造/修改后的 state：load 路径已在
        // BpeState::load_with_options 拒绝过一次，这里是第二道防线）
        reject_sparse_ids(&state.vocab)?;

        // 【新增】：探测最大的 ID，以分配安全大小的数组 (空间换时间)
        // 上一步已保证所有 id ≥ 0 且跨度在预算内 ⟹ `max_id + 1` 不会溢出。
        let max_id = state.vocab.keys().copied().max().unwrap_or(0);
        let array_size = (max_id + 1) as usize;

        let mut fsm_props_array = vec![
            FSMProperties {
                provides: 0,
                hangs: 0
            };
            array_size
        ];
        let mut vocab_strings = vec![String::new(); array_size];

        // info!("成功加载分词器，词表大小: {}, 特殊字符数量: {}", vocab.len(), special_tokens.len());
        let mut v0 = Vec::new();
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        let mut v3 = Vec::new();

        // 1. 遍历基础词表，计算 UTF-8 状态
        for (&id, bytes) in &state.vocab {
            let idx = id as usize;

            // 【预计算并存入 O(1) 数组】
            vocab_strings[idx] = String::from_utf8_lossy(bytes).into_owned();

            let len = bytes.len();
            if len == 0 {
                v0.push(id);
                continue;
            }

            // R4-17：FSM 计算收敛到 `fsm_props_of` 唯一实现（前导字节区间
            // 已排除 C0/C1 overlong 与 F5–F7 超范围，保证"合法"即严格 UTF-8）
            let FSMProperties { provides, hangs } = fsm_props_of(bytes);

            // 【写入新数组】
            fsm_props_array[idx] = FSMProperties { provides, hangs };

            // 预分配到高速查询数组中
            if is_valid_for_pending(provides, len, 0) {
                v0.push(id);
            }
            if is_valid_for_pending(provides, len, 1) {
                v1.push(id);
            }
            if is_valid_for_pending(provides, len, 2) {
                v2.push(id);
            }
            if is_valid_for_pending(provides, len, 3) {
                v3.push(id);
            }
        }

        // 2. 将所有的 Special Tokens 强制加入状态为 0 的合法数组
        // 因为特殊字符 (如 <|im_start|>) 是标准的 ASCII/UTF-8 字符串，既不悬空，也不提供补齐
        for &id in state.special_tokens.values() {
            fsm_props_array[id as usize] = FSMProperties {
                provides: 0,
                hangs: 0,
            };
            v0.push(id);
        }

        // 整理内存，供以后极速借用
        v0.shrink_to_fit();
        v1.shrink_to_fit();
        v2.shrink_to_fit();
        v3.shrink_to_fit();

        info!(
            "FSM 状态机预计算完成。V0:{}, V1:{}, V2:{}, V3:{}",
            v0.len(),
            v1.len(),
            v2.len(),
            v3.len()
        );

        let pre_tokenizer = PreTokenizer::new(&state.pre_tokenizer_pattern)?;

        // 【新增】：提取特殊字符的 ID 到 HashSet
        let mut special_token_ids = HashSet::with_capacity(state.special_tokens.len());
        for &id in state.special_tokens.values() {
            special_token_ids.insert(id);
        }

        Ok(Self {
            pre_tokenizer,
            state,
            special_matcher,
            vocab_strings,   // 注入
            fsm_props_array, // 注入
            valid_for_pending_0: v0,
            valid_for_pending_1: v1,
            valid_for_pending_2: v2,
            valid_for_pending_3: v3,
            special_token_ids,
            memo_owner: next_tokenizer_id(),
        })
    }

    /// 返回"按 merges 数 `k` 截断"的**新**分词器，`self` 不变（等价于 CLI `cut`）。
    ///
    /// 语义与 `BpeState::truncate_to_k` 完全一致（k 超出 K_max 自动钳制），
    /// 并沿用 CLI `cut_loaded` 的 meta 修正规则：`meta.train.actual_merges`
    /// 改写成截断后的事实，`truncated_from` 只在**首次**截断时记录源规模
    /// （连续多次截断保留最初训练规模，而不是上一次的中间值）。
    ///
    /// 之所以返回新实例而非原地修改：模型缓存表（FSM / 接续表 / Aho-Corasick）
    /// 必须与 vocab 同步重建，而 `&mut self` 在共享（`Arc`）场景下不可用。
    pub fn cut_to_merges(&self, k: usize) -> Result<Self, BbpeError> {
        let mut state = self.state.clone();
        let k = state.truncate_to_k(k);
        if let Some(meta) = state.meta.as_mut() {
            if let Some(t) = meta.train.as_mut() {
                t.actual_merges = k;
                if t.truncated_from.is_none() {
                    t.truncated_from = Some(t.vocab_size);
                }
            }
        }
        Self::from_state(state)
    }

    /// 返回"改写预分词 pattern"的**新**分词器，`self` 不变。
    ///
    /// 用途：v0 旧模型迁移（`inspect --migrate --pattern`）与"训练时 pattern
    /// 选错、想在不重训的情况下修正切分口径"的场景。pattern 的**全文覆盖**
    /// 约束不会在这里校验 —— 编码/训练路径上的 `*_checked` 会硬报错。
    pub fn with_pattern(&self, pattern: &str) -> Result<Self, BbpeError> {
        let mut state = self.state.clone();
        state.pre_tokenizer_pattern = pattern.to_string();
        Self::from_state(state)
    }

    pub fn save(&self, filepath: &str) -> Result<(), BbpeError> {
        self.state.save(filepath)
    }

    /// 按保存选项写出模型，见 [`BpeState::save_with_options`]。
    pub fn save_with_options(&self, filepath: &str, opts: SaveOptions) -> Result<(), BbpeError> {
        self.state.save_with_options(filepath, opts)
    }

    // ---------- 新增：带防注入与偏移量追踪的安全外壳 ----------
    //
    // R4-4：`encode` 的历史签名不返回 `Result`，预分词失败（自定义 pattern
    // 未覆盖全部文本）只能 `eprintln!` 后返回**不完整**编码 —— Python 绑定
    // 于是拿到部分 token，`decode(encode(text)) != text`，stderr 告警不能
    // 替代机器可处理错误。现在切分逻辑收敛到 `encode_with_mode` 一份：
    //   - `strict = false`（本方法）：保持历史行为，大声报警 + 尽力返回；
    //   - `strict = true`（[`Self::encode_checked`]）：返回 `Err`，
    //     供训练/评估/绑定层把"整行未被完整切分"变成显式失败。
    pub fn encode(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
        allow_all_special: bool,
    ) -> Encoding {
        // strict = false ⟹ 内部不会返回 Err（错误已就地报警）
        self.encode_with_mode(text, allowed_special, allow_all_special, false)
            .unwrap_or_else(|_| Encoding {
                ids: Vec::new(),
                tokens: Vec::new(),
                offsets: Vec::new(),
            })
    }

    /// `encode` 的**严格版**：预分词失败（正则运行期错误 / 未覆盖间隙）时
    /// 返回 [`BbpeError::PreTokenize`]，而不是返回不完整编码。
    ///
    /// 语义：`Err` 时**没有**可信的部分结果可依赖（已回调的 chunk 虽有效，
    /// 但"本次切分不完整"，调用方应丢弃本轮结果）。需要完整语料语义的调用方
    /// （训练、质量评估、Python 绑定）必须用本方法。
    pub fn encode_checked(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
        allow_all_special: bool,
    ) -> Result<Encoding, BbpeError> {
        self.encode_with_mode(text, allowed_special, allow_all_special, true)
    }

    /// encode 的唯一实现（`strict` 决定预分词错误的处理方式，见 [`Self::encode`]）。
    fn encode_with_mode(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
        allow_all_special: bool,
        strict: bool,
    ) -> Result<Encoding, BbpeError> {
        let mut final_ids = Vec::new();
        let mut final_tokens = Vec::new();
        let mut final_offsets = Vec::new();

        let mut last_end = 0;
        // C4：tiktoken 式约定 —— `allowed_special` 中任一项为 "all" 即放行全部
        // （与 Python 绑定 `resolve_allowed_special` 对齐）。不实现的后果是静默
        // 语义分歧：从 Python 迁接到 Rust 的用户沿用 `{"all"}` 写法会得到一个
        // 空放行集合（所有 special 被切碎）且无任何报错。
        let is_all_allowed = allow_all_special || allowed_special.contains("all");

        // 1. O(N) 极速扫描文本中的所有可能是特殊字符的部分
        for mat in self.special_matcher.find_iter(text) {
            let start = mat.start();
            let end = mat.end();
            // C1 双保险：空 pattern 匹配（start == end）直接跳过。正常加载路径
            // 已在解析期 + V6 双重拒绝空 special token，此处防御未来其他空
            // pattern 来源，保证 encode 永不因非字符边界切片而 panic。
            if start == end {
                continue;
            }
            let match_str = &text[start..end];

            // 2. 将特殊字符左侧的“普通文本”送入你的核心 BPE 引擎
            if start > last_end {
                self.encode_normal_chunk_mode(
                    &text[last_end..start],
                    last_end,
                    &mut final_ids,
                    &mut final_tokens,
                    &mut final_offsets,
                    strict,
                )?;
            }

            // 3. 【核心防线】如果这个特殊字符在白名单里，直接整体切词！
            if is_all_allowed || allowed_special.contains(match_str) {
                let special_id = self.state.special_tokens[match_str];
                final_ids.push(special_id);
                final_tokens.push(match_str.to_string());
                final_offsets.push((start, end));
            } else {
                // 如果是黑客注入的恶意字符（不在白名单），强行当做普通文本送进引擎切碎！
                self.encode_normal_chunk_mode(
                    match_str,
                    start,
                    &mut final_ids,
                    &mut final_tokens,
                    &mut final_offsets,
                    strict,
                )?;
            }
            last_end = end;
        }

        // 4. 处理末尾剩余的普通文本
        if last_end < text.len() {
            self.encode_normal_chunk_mode(
                &text[last_end..],
                last_end,
                &mut final_ids,
                &mut final_tokens,
                &mut final_offsets,
                strict,
            )?;
        }

        Ok(Encoding {
            ids: final_ids,
            tokens: final_tokens,
            offsets: final_offsets,
        })
    }

    /// R4-4：并行路径的任务收集（严格模式记录首个预分词错误，非严格模式大声报警）。
    #[inline(always)]
    fn collect_jobs<'a>(
        &self,
        seg: &'a str,
        base: usize,
        jobs: &mut Vec<ChunkJob<'a>>,
        pretokenize_err: &mut Option<BbpeError>,
        strict: bool,
    ) {
        let result = self
            .pre_tokenizer
            .tokenize_with_checked(seg, |bytes, local_offset| {
                jobs.push(ChunkJob::Normal {
                    bytes,
                    offset: base + local_offset,
                });
            });
        if let Err(e) = result {
            if strict {
                if pretokenize_err.is_none() {
                    *pretokenize_err = Some(e);
                }
            } else {
                crate::pre_tokenizer::report_pretokenize_error(&e);
            }
        }
    }

    // 【参考 Tiktoken 规范】：对于完全不关心 offsets 且拒绝一切特殊字符的快速调用方
    // 提供一个明确表明意图的 encode_ordinary 方法，直接返回 Vec<i32>
    //
    // R4-4 收口（既定语义，勿再“修复”）：本方法是**非严格**快速路径 ——
    // 预分词失败（自定义 pattern 未覆盖全部文本）时大声报警（stderr +
    // tracing::error）后返回**不完整**结果（已切出的 chunk 有效，缺失部分
    // 静默丢弃），签名不返回 Result 是历史 API 约束。需要机器可感知错误的
    // 调用方（Python 绑定、训练/评估）一律用 [`Self::encode_ordinary_checked`]。
    pub fn encode_ordinary(&self, text: &str) -> Vec<i32> {
        let mut ids: Vec<i32> = Vec::new();
        if let Err(e) = self.encode_ordinary_into(text, &mut ids) {
            // R4-4 的既定语义：非严格路径报警后返回**已覆盖的部分**（不是空）
            crate::pre_tokenizer::report_pretokenize_error(&e);
        }
        ids
    }

    /// `encode_ordinary` 的**严格版**（R4-4）：预分词失败返回
    /// [`BbpeError::PreTokenize`]，而不是返回不完整 ids。
    ///
    /// Python 绑定的 `encode_ordinary` 已切换到本方法（与 `encode` 同口径：
    /// 错误映射为异常）；Rust 侧需要“失败可感知”的调用方也应用本方法。
    /// # 为什么不再转调 `encode`
    ///
    /// 原实现是 `self.encode(text, &empty_set, false).ids` —— 走的是完整路径，
    /// **为每个 token 克隆一次字符串并记录字节区间**，最后把 tokens/offsets 整体
    /// 丢弃。10MB 中文实测这部分占掉一半以上时间（2.4M token × String clone）。
    ///
    /// 语义等价性：`allowed_special` 为空时，`encode` 内部对每个 special 命中都走
    /// "按普通文本编码" 的分支（C4 的 all 放行不适用），因此整段扫描 special 与
    /// 直接整段预分词的结果逐 token 相同 —— 已用基线二进制对 10MB/6.8MB/17 行
    /// 边界用例/10 段随机切片做 sha256 差分验证。
    pub fn encode_ordinary_checked(&self, text: &str) -> Result<Vec<i32>, BbpeError> {
        let mut ids: Vec<i32> = Vec::new();
        self.encode_ordinary_into(text, &mut ids)?;
        Ok(ids)
    }

    /// 裸 ID 路径的唯一实现：出错时 `ids` 保留**已覆盖的前缀**（非严格调用方依赖此语义）。
    #[inline]
    fn encode_ordinary_into(&self, text: &str, ids: &mut Vec<i32>) -> Result<(), BbpeError> {
        self.pre_tokenizer
            .tokenize_with_checked(text, |chunk, _off| {
                self.bpe_core_ordinary(chunk, ids);
            })
    }

    // ---------- 全功能长文本并行护城河 ----------
    pub fn encode_parallel(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
        allow_all_special: bool,
        num_threads: Option<usize>,
    ) -> Encoding {
        // R4-4：与 encode 同构 —— 非严格模式大声报警后尽力返回
        self.encode_parallel_with_mode(text, allowed_special, allow_all_special, num_threads, false)
            .unwrap_or_else(|_| Encoding {
                ids: Vec::new(),
                tokens: Vec::new(),
                offsets: Vec::new(),
            })
    }

    /// `encode_parallel` 的**严格版**（R4-4）：预分词失败返回 `Err`，
    /// 与 [`Self::encode_checked`] 同语义。
    pub fn encode_parallel_checked(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
        allow_all_special: bool,
        num_threads: Option<usize>,
    ) -> Result<Encoding, BbpeError> {
        self.encode_parallel_with_mode(text, allowed_special, allow_all_special, num_threads, true)
    }

    fn encode_parallel_with_mode(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
        allow_all_special: bool,
        num_threads: Option<usize>,
        strict: bool,
    ) -> Result<Encoding, BbpeError> {
        let mut jobs = Vec::new();
        // R4-4：严格模式下记住首个预分词错误（收集任务的主线程无 `?` 可用）
        let mut pretokenize_err: Option<BbpeError> = None;
        let mut last_end = 0;
        // C4：与 encode 同口径的 "all" 约定（详见 encode 内注释）
        let is_all_allowed = allow_all_special || allowed_special.contains("all");

        // 1. 极速扫描收集任务 (主线程 O(N) 极低开销)
        for mat in self.special_matcher.find_iter(text) {
            let start = mat.start();
            let end = mat.end();
            // C1 双保险：空 pattern 匹配（start == end）直接跳过，理由同 encode
            if start == end {
                continue;
            }
            let match_str = &text[start..end];

            if start > last_end {
                self.collect_jobs(
                    &text[last_end..start],
                    last_end,
                    &mut jobs,
                    &mut pretokenize_err,
                    strict,
                );
            }

            if is_all_allowed || allowed_special.contains(match_str) {
                let id = self.state.special_tokens[match_str];
                jobs.push(ChunkJob::Special {
                    id,
                    token_str: match_str,
                    offset: (start, end),
                });
            } else {
                self.collect_jobs(match_str, start, &mut jobs, &mut pretokenize_err, strict);
            }
            last_end = end;
        }

        if last_end < text.len() {
            self.collect_jobs(
                &text[last_end..],
                last_end,
                &mut jobs,
                &mut pretokenize_err,
                strict,
            );
        }

        if let Some(e) = pretokenize_err {
            return Err(e);
        }

        // 2. 动态调度阈值：保护短对话的绝对极速，赋予长文本的多核算力
        if text.len() < 5000 {
            let mut enc = Encoding {
                ids: Vec::new(),
                tokens: Vec::new(),
                offsets: Vec::new(),
            };
            for job in jobs {
                match job {
                    ChunkJob::Special {
                        id,
                        token_str,
                        offset,
                    } => {
                        enc.ids.push(id);
                        enc.tokens.push(token_str.to_string());
                        enc.offsets.push(offset);
                    }
                    ChunkJob::Normal { bytes, offset } => {
                        self.bpe_core_full(
                            bytes,
                            offset,
                            &mut enc.ids,
                            &mut enc.tokens,
                            &mut enc.offsets,
                        );
                    }
                }
            }
            Ok(enc)
        } else {
            // 【核心注入】：将 Rayon 的并发逻辑包在一个闭包里
            let execute_parallel = || {
                jobs.par_iter()
                    .fold(
                        || Encoding {
                            ids: Vec::new(),
                            tokens: Vec::new(),
                            offsets: Vec::new(),
                        },
                        |mut acc, job| {
                            match job {
                                ChunkJob::Special {
                                    id,
                                    token_str,
                                    offset,
                                } => {
                                    acc.ids.push(*id);
                                    acc.tokens.push(token_str.to_string());
                                    acc.offsets.push(*offset);
                                }
                                ChunkJob::Normal { bytes, offset } => {
                                    self.bpe_core_full(
                                        bytes,
                                        *offset,
                                        &mut acc.ids,
                                        &mut acc.tokens,
                                        &mut acc.offsets,
                                    );
                                }
                            }
                            acc
                        },
                    )
                    .reduce(
                        || Encoding {
                            ids: Vec::new(),
                            tokens: Vec::new(),
                            offsets: Vec::new(),
                        },
                        |mut a, b| {
                            a.ids.extend(b.ids);
                            a.tokens.extend(b.tokens);
                            a.offsets.extend(b.offsets);
                            a
                        },
                    )
            };

            // 【资源控制分发】：按需构建局部线程池或使用全局线程池
            Ok(match num_threads {
                Some(n) if n > 0 => {
                    // 构建一个临时的受限线程池
                    match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
                        // install 会阻塞当前线程，但在池中受限执行
                        Ok(pool) => pool.install(execute_parallel),
                        Err(_) => execute_parallel(), // 若构建失败（极罕见），降级回全局池
                    }
                }
                _ => execute_parallel(), // None 或 0 时，火力全开，使用 Rayon 全局池
            })
        }
    }

    // ---------- 极致裸 ID 并行引擎 (SFT 训练标签生成神技) ----------
    //
    // R4-4 收口：与 encode_ordinary 同语义 —— **非严格**快速路径，预分词失败
    // 大声报警后返回不完整结果；需要硬失败的调用方用
    // [`Self::encode_ordinary_parallel_checked`]。两者的任务执行逻辑共用
    // `execute_ordinary_jobs`（与 m2 同一种修法：消除复制，而非就地修补）。
    pub fn encode_ordinary_parallel(&self, text: &str, num_threads: Option<usize>) -> Vec<i32> {
        // 直接绕过所有的特殊字符校验，调用预分词器生成只包含 Normal 字节的任务
        let chunks = self.pre_tokenizer.pre_tokenize(text);
        self.execute_ordinary_jobs(chunks, text.len(), num_threads)
    }

    /// `encode_ordinary_parallel` 的**严格版**（R4-4）：预分词失败返回 `Err`，
    /// 与 [`Self::encode_ordinary_checked`] 同语义。Python 绑定的
    /// `encode_ordinary_parallel` 已切换到本方法。
    pub fn encode_ordinary_parallel_checked(
        &self,
        text: &str,
        num_threads: Option<usize>,
    ) -> Result<Vec<i32>, BbpeError> {
        let mut chunks = Vec::new();
        self.pre_tokenizer
            .tokenize_with_checked(text, |bytes, offset| chunks.push((bytes, offset)))?;
        Ok(self.execute_ordinary_jobs(chunks, text.len(), num_threads))
    }

    /// 裸 ID 任务的唯一执行实现（串行/并行分发与资源控制）。
    fn execute_ordinary_jobs(
        &self,
        chunks: Vec<(&[u8], usize)>,
        text_len: usize,
        num_threads: Option<usize>,
    ) -> Vec<i32> {
        if text_len < 5000 {
            let mut final_ids = Vec::with_capacity(text_len / 2);
            for (bytes, _) in chunks {
                self.bpe_core_ordinary(bytes, &mut final_ids);
            }
            final_ids
        } else {
            let execute_parallel = || {
                chunks
                    .par_iter()
                    .fold(Vec::new, |mut acc, &(bytes, _)| {
                        self.bpe_core_ordinary(bytes, &mut acc);
                        acc
                    })
                    .reduce(Vec::new, |mut a, b| {
                        a.extend(b);
                        a
                    })
            };

            // 【资源控制分发】
            match num_threads {
                Some(n) if n > 0 => match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
                    Ok(pool) => pool.install(execute_parallel),
                    Err(_) => execute_parallel(),
                },
                _ => execute_parallel(),
            }
        }
    }

    // ---------- BPE 合并核心（唯一实现）----------
    //
    // m2：历史上 `bpe_core_full` / `bpe_core_ordinary` / `encode_normal_chunk`
    // 各持一份 ~110 行的复制实现。任何一处语义修改都必须同步三处，否则串行
    // `encode` 与并行 `encode_parallel` 会**静默分叉**（只有 T-8 能发现）。
    // 此处统一为一份实现，三个入口全部退化为薄封装。
    //
    // `track` 决定收集粒度：
    //   - `None`             ⟹ 只产出 id（极速裸 ID 路径：不建偏移表、不查字符串）
    //   - `Some((tok, off))` ⟹ 额外产出 token 字符串与字节偏移
    //
    // 偏移继承语义（与合并前三份实现逐条一致）：合并时左侧节点吞噬右侧节点的
    // 结束偏移，即 `off[left].1 = off[right].1`。
    #[inline(always)]
    fn bpe_merge_core(
        &self,
        bytes_chunk: &[u8],
        global_offset: usize,
        final_ids: &mut Vec<i32>,
        track: Option<Track<'_>>,
    ) {
        // ---- 极短 chunk 快速路径（中文语料里大量 1~2 字节的空白/标点/数字）----
        match bytes_chunk.len() {
            0 => return,
            1 => {
                let id = bytes_chunk[0] as i32;
                final_ids.push(id);
                if let Some((tok, off)) = track {
                    tok.push(self.token_string(id));
                    off.push((global_offset, global_offset + 1));
                }
                return;
            }
            2 => {
                let (a, b) = (bytes_chunk[0] as i32, bytes_chunk[1] as i32);
                match self.state.merge_lookup(a, b) {
                    Some(merged) => {
                        final_ids.push(merged);
                        if let Some((tok, off)) = track {
                            tok.push(self.token_string(merged));
                            off.push((global_offset, global_offset + 2));
                        }
                    }
                    None => {
                        final_ids.push(a);
                        final_ids.push(b);
                        if let Some((tok, off)) = track {
                            tok.push(self.token_string(a));
                            tok.push(self.token_string(b));
                            off.push((global_offset, global_offset + 1));
                            off.push((global_offset + 1, global_offset + 2));
                        }
                    }
                }
                return;
            }
            _ => {}
        }

        // 常规路径：先查 chunk 缓存（仅裸 ID 路径），未命中再借用工作区合并
        let use_memo = track.is_none() && bytes_chunk.len() <= MEMO_MAX_CHUNK;
        if use_memo {
            let mut hit = false;
            CHUNK_MEMO.with(|cell| {
                let memo = cell.borrow();
                if memo.owner == self.memo_owner {
                    if let Some(ids) = memo.map.get(bytes_chunk) {
                        final_ids.extend_from_slice(ids);
                        hit = true;
                    }
                }
            });
            if hit {
                return;
            }
        }

        let start = final_ids.len();
        BPE_SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            self.bpe_merge_core_inner(&mut guard, bytes_chunk, global_offset, final_ids, track);
        });

        if use_memo {
            CHUNK_MEMO.with(|cell| {
                let mut memo = cell.borrow_mut();
                if memo.owner != self.memo_owner {
                    // 换了模型：旧缓存作废，避免把别的词表结果当命中
                    memo.map.clear();
                    memo.owner = self.memo_owner;
                }
                if memo.map.len() < MEMO_CAP {
                    memo.map
                        .insert(bytes_chunk.into(), final_ids[start..].into());
                }
            });
        }
    }

    /// BPE 合并主体。`scratch` 由调用方提供并复用（热路径零分配）。
    #[inline(always)]
    fn bpe_merge_core_inner(
        &self,
        scratch: &mut BpeScratch,
        bytes_chunk: &[u8],
        global_offset: usize,
        final_ids: &mut Vec<i32>,
        mut track: Option<Track<'_>>,
    ) {
        let BpeScratch {
            ids,
            prev,
            next,
            deleted,
            offsets,
            heap,
        } = scratch;

        ids.clear();
        ids.extend(bytes_chunk.iter().map(|&b| b as i32));
        let len = ids.len();

        // 偏移表仅在需要追踪时构建 ⟹ 裸 ID 路径零额外开销
        offsets.clear();
        if track.is_some() {
            offsets.extend((0..len).map(|i| (global_offset + i, global_offset + i + 1)));
        }

        prev.clear();
        prev.extend((0..len).map(|i| i as isize - 1));
        next.clear();
        next.extend((0..len).map(|i| i as isize + 1));
        deleted.clear();
        deleted.resize(len, false);
        heap.clear();

        for i in 0..len - 1 {
            if let Some(rank) = self.state.merge_lookup(ids[i], ids[i + 1]) {
                heap.push(MergeItem {
                    rank,
                    pos: i,
                    id0: ids[i],
                    id1: ids[i + 1],
                });
            }
        }

        while let Some(item) = heap.pop() {
            let left_idx = item.pos;
            if deleted[left_idx] {
                continue;
            }
            let right_idx = next[left_idx] as usize;
            if right_idx >= len || deleted[right_idx] {
                continue;
            }
            if ids[left_idx] != item.id0 || ids[right_idx] != item.id1 {
                continue;
            }

            ids[left_idx] = item.rank;
            deleted[right_idx] = true;
            // 偏移继承必须作用在**本 chunk 的局部偏移表**上：此时 final_offsets
            // 尚未写入本 chunk 的任何条目（收集在循环结束后统一 push），若写
            // final_offsets 则会索引到上一个 chunk 的残留数据甚至越界 panic。
            if track.is_some() {
                offsets[left_idx].1 = offsets[right_idx].1;
            }

            let new_right_idx = next[right_idx];
            next[left_idx] = new_right_idx;
            if (new_right_idx as usize) < len {
                prev[new_right_idx as usize] = left_idx as isize;
            }

            let prev_idx = prev[left_idx];
            if prev_idx >= 0 {
                let p = prev_idx as usize;
                if !deleted[p] {
                    if let Some(rank) = self.state.merge_lookup(ids[p], ids[left_idx]) {
                        heap.push(MergeItem {
                            rank,
                            pos: p,
                            id0: ids[p],
                            id1: ids[left_idx],
                        });
                    }
                }
            }

            if (new_right_idx as usize) < len {
                let n = new_right_idx as usize;
                if !deleted[n] {
                    if let Some(rank) = self.state.merge_lookup(ids[left_idx], ids[n]) {
                        heap.push(MergeItem {
                            rank,
                            pos: left_idx,
                            id0: ids[left_idx],
                            id1: ids[n],
                        });
                    }
                }
            }
        }

        let mut curr = 0isize;
        while curr < len as isize {
            if !deleted[curr as usize] {
                let id = ids[curr as usize];
                final_ids.push(id);
                if let Some((tok, off)) = track.as_mut() {
                    tok.push(self.token_string(id));
                    off.push(offsets[curr as usize]);
                }
            }
            curr = next[curr as usize];
        }
    }

    /// O(1) 取 token 的字符串表示。
    ///
    /// nit2：原 `encode_normal_chunk` 用 `self.state.vocab[&id]` 直接索引 HashMap
    /// 再 `from_utf8_lossy` —— 既慢（每次哈希查找，而预计算数组是 O(1) 下标），
    /// 又会在 id 缺失时 panic。改为优先查预计算数组，缺失回退 vocab 表，再缺失返回空串。
    /// 注意用 `usize::try_from(id)` 而非 `id as usize`：后者对负 id 会得到巨值。
    #[inline(always)]
    fn token_string(&self, id: i32) -> String {
        if let Some(s) = usize::try_from(id)
            .ok()
            .and_then(|i| self.vocab_strings.get(i))
        {
            if !s.is_empty() {
                return s.clone();
            }
        }
        self.state
            .vocab
            .get(&id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default()
    }

    // ---------- BPE 核心：全记录版本 ----------
    #[inline(always)]
    fn bpe_core_full(
        &self,
        bytes_chunk: &[u8],
        global_offset: usize,
        final_ids: &mut Vec<i32>,
        final_tokens: &mut Vec<String>,
        final_offsets: &mut Vec<(usize, usize)>,
    ) {
        self.bpe_merge_core(
            bytes_chunk,
            global_offset,
            final_ids,
            Some((final_tokens, final_offsets)),
        );
    }

    // ---------- BPE 核心：极速裸 ID 版本 ----------
    #[inline(always)]
    fn bpe_core_ordinary(&self, bytes_chunk: &[u8], final_ids: &mut Vec<i32>) {
        // m2：track = None ⟹ 只产出 id，不建偏移表、不查字符串
        self.bpe_merge_core(bytes_chunk, 0, final_ids, None);
    }

    // ---------- 串行 encode 的普通文本分片入口 ----------
    #[inline(always)]
    fn encode_normal_chunk_mode(
        &self,
        text_chunk: &str,
        global_offset: usize,
        final_ids: &mut Vec<i32>,
        final_tokens: &mut Vec<String>,
        final_offsets: &mut Vec<(usize, usize)>,
        strict: bool,
    ) -> Result<(), BbpeError> {
        // m2：原实现在此处内联了第三份 BPE 合并循环（~110 行），并用
        // `self.state.vocab[&id]` 做哈希查找。现在只负责预分词 + 偏移换算，
        // 合并逻辑统一走 `bpe_merge_core`（与并行路径同源，杜绝静默分叉）。
        //
        // R4-4：`strict` 决定预分词错误的去向 —— true 时向上传播（调用方拿到
        // 机器可处理的 Err）；false 时保持历史行为（`eprintln!` 大声报警后
        // 丢弃该分片剩余部分，尽力返回）。
        let result =
            self.pre_tokenizer
                .tokenize_with_checked(text_chunk, |bytes_chunk, local_offset| {
                    let chunk_global_start = global_offset + local_offset;
                    self.bpe_merge_core(
                        bytes_chunk,
                        chunk_global_start,
                        final_ids,
                        Some((final_tokens, final_offsets)),
                    );
                });
        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                if strict {
                    Err(e)
                } else {
                    crate::pre_tokenizer::report_pretokenize_error(&e);
                    Ok(())
                }
            }
        }
    }

    pub fn decode(&self, ids: &[i32]) -> Result<String, BbpeError> {
        let mut res_bytes = Vec::new();
        for &id in ids {
            if let Some(bytes) = self.state.vocab.get(&id) {
                res_bytes.extend(bytes);
            } else {
                return Err(BbpeError::InvalidTokenId(id));
            }
        }
        Ok(String::from_utf8_lossy(&res_bytes).into_owned())
    }

    // A8：以下三个符号已删除 —— `encode_with_types` / `encode_with_types_batch`
    //     / `SftEncoding` / `TokenType`。
    //
    // 原因（全仓 grep 确认：引用完全封闭在本文件内，无任何外部调用者 ——
    // 未导出到 Python、CLI 未用、无测试覆盖）：
    //
    // 1. **零调用者**。它是"供 SFT 掩码引擎使用"的半成品，但 SFT 掩码最终在
    //    Python 侧实现（`python/bbpe_tokenizer/core.py:encode_dialogue_safely`）。
    //
    // 2. **两者语义不同，Rust 版不能替代 Python 版**。Rust 版按 **token 类型**
    //    （Normal/Special）标注；Python 版按 **对话轮次角色**（assistant 轮才
    //    计入 labels）生成 `-100` 掩码。SFT 的正确语义是后者：若按 token 类型
    //    掩码，user 轮里出现的 special token 会被当成"可学习"，从而把 user
    //    内容泄漏进 labels。故 Rust 版并非"更好的实现"，而是**被废弃的替代方案**。
    //
    // 3. 两份逻辑并存本身就是分叉隐患 —— 未来改 SFT 语义时极易改错那一份。
    //
    // 保留 `encode_ordinary_batch`（megadata_rs 实际在用），删除的只是
    // encode_with_types / encode_with_types_batch / SftEncoding / TokenType。

    // ==========================================
    // 接口 2 & 3：零开销的多线程批处理 (Rayon 驱动)
    // ==========================================
    /// 预训练极速裸 ID 批处理：megadata_rs 喂入切片，返回巨型 ID 二维数组
    pub fn encode_ordinary_batch<'a, I>(&self, texts: I) -> Vec<Vec<i32>>
    where
        I: IntoParallelIterator<Item = &'a str>,
    {
        texts
            .into_par_iter()
            // 内部直接调用我们早写好的纯享版极速编码
            .map(|text| self.encode_ordinary(text))
            .collect()
    }

    // ==========================================
    // 接口 4：FSM 探测 API 与 特殊字符探测
    // ==========================================
    /// 【极速探测】：O(1) 判定该 ID 是否属于特殊控制字符
    #[inline(always)]
    pub fn is_special_id(&self, token_id: i32) -> bool {
        self.special_token_ids.contains(&token_id)
    }

    /// 【极速防线】：探测当前 Token 在进行长序列 Packing 截断时是否会导致乱码
    /// 结合预计算的 hangs (悬空字节)，精准拦截把半个汉字切断的惨剧
    #[inline(always)]
    pub fn is_safe_to_truncate_after(&self, token_id: i32) -> bool {
        if token_id < 0 {
            return true;
        }
        let idx = token_id as usize;
        if idx < self.fsm_props_array.len() {
            // 如果 hangs == 0，说明这个 Token 结尾是干净的，可以直接截断
            self.fsm_props_array[idx].hangs == 0
        } else {
            true // 未知词 / 特殊控制符默认安全
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsm_utf8_properties() {
        // 我们不加载文件，直接手动模拟几个典型 Token 测试 FSM 算法
        //
        // R4-17：测试改用与 load 共用的 `fsm_props_of` —— 原先这里另抄一份
        // 计算逻辑，"前导字节区间修正"只改了 load 一份，测试继续按旧区间
        // 放行 C0/C1/F5–F7，属于"改了代码没改测试"的漂移形态。

        // 案例 1：纯英文，不悬空不提供
        let p1 = fsm_props_of(b"hello");
        assert_eq!(p1.provides, 0);
        assert_eq!(p1.hangs, 0);

        // 案例 2：汉字“中”的第一个字节 0xE4 (11100100)，需要 2 个后续字节
        let p2 = fsm_props_of(&[0xE4]);
        assert_eq!(p2.provides, 0);
        assert_eq!(p2.hangs, 2);

        // 案例 3：汉字“中”的后两个字节 0xB8 0xAD (10xxxxxx)，提供了 2 个延续字节
        let p3 = fsm_props_of(&[0xB8, 0xAD]);
        assert_eq!(p3.provides, 2);
        assert_eq!(p3.hangs, 0);

        // 案例 4：坏死重置（先来一个 3 字节头，但中间断了，又来一个 2 字节头）
        let p4 = fsm_props_of(&[0xE4, 0xC2]); // C2 是 11000010，悬空 1
        assert_eq!(p4.provides, 0);
        assert_eq!(p4.hangs, 1, "坏死恢复计算错误");

        // 案例 5（R4-17）：非法前导字节必须**不开启悬空**，否则 FSM 会把
        // overlong / 超范围序列当合法字符放行（"严格 UTF-8"名不副实）
        for b in [0xC0u8, 0xC1, 0xF5, 0xF6, 0xF7, 0x80, 0xFF] {
            assert_eq!(
                fsm_props_of(&[b]).hangs,
                0,
                "字节 {b:#04x} 不是合法 UTF-8 前导"
            );
        }
        // 合法边界值：C2 是最小合法 2 字节头，F4 是最大合法 4 字节头
        assert_eq!(fsm_props_of(&[0xC2]).hangs, 1);
        assert_eq!(fsm_props_of(&[0xF4]).hangs, 3);
    }

    // R4-17（第五轮补）：序列级断言 —— 评审点名的三个非法序列。
    //
    // 保证边界已收敛为“前导字节级”（见 utf8_lead_hangs 文档）：
    // - C0/C1 开头的 overlong 序列：非法前导**不开启悬空**，FSM 不再把
    //   它当“合法 2 字节字符”跟踪（旧实现 C0 会开 hangs=1，80 补齐后
    //   以“合法字符”身份通过掩码）；
    // - surrogate / 超范围标量：前导本身合法（ED/F4），计数层面被当作
    //   完整字符消费（hangs 归零）—— FSM **不拒绝**它们，与 docstring
    //   声明一致。本测试把该边界钉死：未来若升级为序列级校验，这里会
    //   显式失败提醒同步更新所有声明处，而不是静默漂移。
    #[test]
    fn r4_17_sequence_level_matches_leading_byte_contract() {
        // C0 80（overlong U+0000）：C0 非法前导 ⟹ 不开启悬空；
        // 80 是孤立延续字节，也不开启悬空 ⟹ (0, 0)
        let p = fsm_props_of(&[0xC0, 0x80]);
        assert_eq!(
            (p.provides, p.hangs),
            (0, 0),
            "C0 80 不得被当作合法 2 字节字符跟踪"
        );
        // C1 BF（overlong 上界）同理
        assert_eq!(fsm_props_of(&[0xC1, 0xBF]).hangs, 0);

        // ED A0 80（surrogate U+D800）：前导合法，计数层面消费完整（2→1→0）
        let s = fsm_props_of(&[0xED, 0xA0, 0x80]);
        assert_eq!(
            (s.provides, s.hangs),
            (0, 0),
            "surrogate 在计数层面被消费（FSM 不拒绝，已如实声明）"
        );

        // F4 90 80 80（> U+10FFFF）：同上（3→2→1→0）
        let o = fsm_props_of(&[0xF4, 0x90, 0x80, 0x80]);
        assert_eq!(
            (o.provides, o.hangs),
            (0, 0),
            "超范围标量在计数层面被消费（FSM 不拒绝，已如实声明）"
        );

        // 对照：合法序列同样消费完整（证明断言不是“凡多字节就拒绝”）
        assert_eq!(fsm_props_of(&[0xC2, 0xA2]).hangs, 0); // ¢ U+00A2
        assert_eq!(fsm_props_of("中".as_bytes()).hangs, 0);

        // 关键差异在“孤立前导是否开启悬空”：合法前导开，非法前导不开
        assert_eq!(fsm_props_of(&[0xED]).hangs, 2);
        assert_eq!(fsm_props_of(&[0xF4]).hangs, 3);
        assert_eq!(fsm_props_of(&[0xC0]).hangs, 0);
        assert_eq!(fsm_props_of(&[0xF5]).hangs, 0);
    }
}

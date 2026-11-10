use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
// use rustc_hash::FxHashMap;
use ahash::AHashMap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// 【已清理】去掉了多余的 BufRead, BufReader
use crate::CorpusReader;
use crate::cancel::{CancelReason, CancelToken};
use crate::error::BbpeError;
use crate::pre_tokenizer::PreTokenizer;
use tracing::info;

// n4：原先 `use better_mimalloc_rs::MiMalloc;` 与下面的 `#[global_allocator]`
// 均已被注释，mimalloc 从未真正启用 —— 依赖与导入双双是死代码。
// 若日后要启用，需恢复本导入 + 取消注释 `#[global_allocator]`（会改变全局
// 分配器行为，属独立决策，不在本次评审修复范围内），并重新加回 Cargo 依赖。
use crate::reader::CorpusStats;
use crate::tokenizer::{BpeState, SaveOptions, TrainMeta};
use std::sync::Arc;

// #[global_allocator]
// static GLOBAL: MiMalloc = MiMalloc;

// ==========================================
// 核心优化 1：位压缩单指令哈希助手函数
// ==========================================
#[inline(always)]
fn record_change(deltas: &mut AHashMap<u64, i64>, pair: u64, change: i64) {
    if change == 0 {
        return;
    }
    *deltas.entry(pair).or_insert(0) += change;
}

#[inline(always)]
fn pack_pair(p0: u32, p1: u32) -> u64 {
    ((p0 as u64) << 32) | (p1 as u64)
}

#[inline(always)]
fn unpack_pair(val: u64) -> (u32, u32) {
    ((val >> 32) as u32, (val & 0xFFFFFFFF) as u32)
}

/// M5：把 `usize` 收敛为 `u32`，**越界即报错而非静默截断**。
///
/// Shard 的扁平存储（`flat_data` 偏移 / `word_lens` / 倒排链表下标）全链路用 u32
/// 索引以省内存。原先写作 `x as u32`，一旦某个 shard 的数据量超过 4GiB 就会静默
/// 回绕 —— 后续所有 `data[start..start+len]` 读到错误位置，产出**看似正常但词频
/// 全错**的模型，且无任何报错。这里改为显式失败，并给出可操作的处置建议。
#[inline(always)]
fn u32_checked(v: usize, what: &str) -> Result<u32, BbpeError> {
    u32::try_from(v).map_err(|_| {
        BbpeError::CapacityExceeded(format!(
            "{what} = {v} 超出 u32 上限 {}。单个 Shard 的数据量过大，\
             请增大 `RuntimeConfig::num_shards` 以摊薄，或降低语料规模",
            u32::MAX
        ))
    })
}

/// B3 辅助：紧凑化 flat_data，丢弃所有"pair 全 < m"的冻结 word。
/// active_pairs = global_counts 中频次 >= m 的 pair 集合。
/// word 的所有 pair 都不在 active_pairs → 冻结(丢弃字节内容)。
/// 返回 (new_data, new_starts, new_lens, new_weights)。
/// A1(clean)：四元组过长，抽出类型别名（clippy::type_complexity）。
type Compacted<T> = (Vec<T>, Vec<u32>, Vec<u32>, Vec<u64>);
fn compact_freeze<T: Copy>(
    data: &[T],
    word_starts: &[u32],
    word_lens: &[u32],
    word_weights: &[u64],
    global_counts: &AHashMap<u64, u64>,
    m: u64,
    conv: impl Fn(T) -> u32,
) -> Result<Compacted<T>, BbpeError> {
    let mut new_data: Vec<T> = Vec::new();
    let mut new_starts = Vec::with_capacity(word_starts.len());
    let mut new_lens = Vec::with_capacity(word_lens.len());
    let mut new_weights = Vec::with_capacity(word_weights.len());
    for i in 0..word_starts.len() {
        let start = word_starts[i] as usize;
        let len = word_lens[i] as usize;
        // 冻结判定：word 所有 pair 都不在 active_pairs (即全局频次 < m)
        let frozen = if len < 2 {
            true // 无 pair，不参与合并，冻结省字节
        } else {
            let mut fr = true;
            for j in 0..len - 1 {
                let pair = pack_pair(conv(data[start + j]), conv(data[start + j + 1]));
                if global_counts.get(&pair).copied().unwrap_or(0) >= m {
                    fr = false;
                    break;
                }
            }
            fr
        };
        if !frozen {
            let ns = new_data.len();
            new_data.extend_from_slice(&data[start..start + len]);
            // M5：紧凑化后的偏移/长度索引越界检查
            new_starts.push(u32_checked(ns, "紧凑化后 flat_data 偏移")?);
            new_lens.push(u32_checked(len, "word 长度")?);
            new_weights.push(word_weights[i]);
        }
    }
    Ok((new_data, new_starts, new_lens, new_weights))
}

// ==========================================
// 核心优化 2：懒惰大顶堆节点 (Lazy Priority Queue)
// ==========================================
#[derive(Copy, Clone, Eq, PartialEq)]
struct HeapEntry {
    count: u64,
    pair: u64,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            // count 相等时，按字典序给较小的 pair 赋予更高优先级
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ==========================================
// 核心优化 3：双态扁平存储 (初始 u8 紧凑，按需升级 u32)
// 初始阶段所有 token 均为字节 (0-255)，用 u8 存储省 4x 内存；
// 当首次写入 ≥256 的 merge id 时，整个 Shard 一次性升级为 u32。
// ==========================================
#[derive(Clone)]
enum FlatData {
    Bytes(Vec<u8>),
    Wide(Vec<u32>),
}

impl FlatData {
    #[inline(always)]
    fn reserve_exact(&mut self, additional: usize) {
        match self {
            FlatData::Bytes(b) => b.reserve_exact(additional),
            FlatData::Wide(w) => w.reserve_exact(additional),
        }
    }
}

// ---------- 冻结 GC 的自适应间隔策略（2026-11-05）----------
//
// 语义前提：`gc_freeze` 只删除"所有 pair 频次 < m"的 word —— 这类 word 永远
// 不可能再被选中，删掉不改变任何一次 merge 的结果，只影响内存回收时机。
// 因此"多久跑一次"是纯性能/内存权衡，可以自由调整（由 tests::gc_policy 守护）。
//
// 依据（9.96GB 语料剖析）：每次 gc_freeze 都要重建**全量倒排索引**（1.89 亿
// unique word），而 m=2 时前期只冻结 309 个 word，固定每 500 merges 跑一次
// 等于把 59% 的时间扔在无效重建上。
pub(crate) const GC_INTERVAL_BASE: usize = 500;
pub(crate) const GC_INTERVAL_MAX: usize = 20000;
/// 低于此冻结产率 ⟹ 认为"没什么可回收"，间隔翻倍。
pub(crate) const GC_LOW_YIELD: f64 = 0.003;
/// 高于此冻结产率 ⟹ 认为"死词堆积快"，回到基线密集回收。
pub(crate) const GC_HIGH_YIELD: f64 = 0.01;

/// 根据上一轮冻结产率决定下一轮 GC 间隔（纯函数，便于单测）。
#[inline]
pub(crate) fn next_gc_interval(prev: usize, frozen_ratio: f64) -> usize {
    if frozen_ratio < GC_LOW_YIELD {
        prev.saturating_mul(2).min(GC_INTERVAL_MAX)
    } else if frozen_ratio > GC_HIGH_YIELD {
        GC_INTERVAL_BASE
    } else {
        prev.max(1) // 中间带：保持当前节奏（防御性 max(1)，避免 0 间隔死循环）
    }
}

// ==========================================
// 扁平化 Shard (基于 SIMD 寻址 & 双指针)
// ==========================================
#[derive(Clone)]
struct Shard {
    // Arena 内存池：所有词的数据全部挤在这一个连续的扁平大数组里
    flat_data: FlatData,
    word_starts: Vec<u32>,
    word_lens: Vec<u32>, // 使用 u32 防止遇到畸形超长无空格文本
    // 频次全链路 u64（方案 §6.4/F12）：数百 GB 语料下高频词频次 ~10^9-10^10 必然溢出 u32
    word_weights: Vec<u64>,

    // pair_head 替代 local_counts：判断 shard 是否含某 pair 同时提供倒排链表
    pair_head: AHashMap<u64, u32>, // Pair -> 链表头节点在 node_word 中的索引
    node_word: Vec<u32>,           // 存储词的 idx
    node_next: Vec<u32>,           // 存储下一个节点的索引 (u32::MAX 表示链表结束)

    // 【新增】：复用缓冲区
    merge_buffer: Vec<u32>,
}

impl Shard {
    fn new() -> Self {
        Self {
            flat_data: FlatData::Bytes(Vec::new()),
            word_starts: Vec::new(),
            word_lens: Vec::new(),
            word_weights: Vec::new(),

            pair_head: AHashMap::default(),
            node_word: Vec::new(),
            node_next: Vec::new(),
            merge_buffer: Vec::with_capacity(2048),
        }
    }

    fn add_chunk(&mut self, chunk_bytes: &[u8], count: u64) -> Result<(), BbpeError> {
        // 装载阶段所有值均为字节 (0-255)，用 u8 紧凑存储
        let bytes = match &mut self.flat_data {
            FlatData::Bytes(b) => b,
            FlatData::Wide(_) => {
                unreachable!("add_chunk 仅在初始装载阶段调用，此时 flat_data 必为 Bytes")
            }
        };
        // M5：`as u32` → 显式检查。start 是该 word 在 shard 内 flat_data 的字节偏移，
        // 单 shard 累计超 4GiB 时原实现会静默回绕。
        let start = u32_checked(bytes.len(), "shard flat_data 字节偏移")?;
        let len = u32_checked(chunk_bytes.len(), "chunk 字节长度")?;
        bytes.extend_from_slice(chunk_bytes);
        self.word_starts.push(start);
        self.word_lens.push(len);
        self.word_weights.push(count);
        Ok(())
    }

    fn build_index(&mut self) -> Result<(), BbpeError> {
        // 【精准算命】：不要猜！直接遍历 word_lens，算出绝对精准的 Pair 出现总次数
        let mut exact_max_pairs = 0;
        for &len in &self.word_lens {
            exact_max_pairs += len.saturating_sub(1) as usize;
        }

        // reserve_exact 保证这两个数组分配后一次都不扩容！
        self.node_word.reserve_exact(exact_max_pairs);
        self.node_next.reserve_exact(exact_max_pairs);

        // HashMap 保守起见给同样的容量，配合随后的 shrink_to_fit 完美收缩
        // self.local_counts.reserve(exact_max_pairs);
        // self.pair_head.reserve(exact_max_pairs);

        self.pair_head.reserve(50_000);

        // build_index 阶段 flat_data 必为 Bytes (尚未发生任何 merge)
        let bytes = match &self.flat_data {
            FlatData::Bytes(b) => b,
            FlatData::Wide(_) => unreachable!("build_index 阶段 flat_data 必为 Bytes"),
        };

        for i in 0..self.word_starts.len() {
            let start = self.word_starts[i] as usize;
            let len = self.word_lens[i] as usize;

            let slice = &bytes[start..start + len];
            for w in slice.windows(2) {
                let pair = pack_pair(w[0] as u32, w[1] as u32);

                // 【优化：O(1) 数组链表头插法】
                let head = self.pair_head.entry(pair).or_insert(u32::MAX);
                // 防御重复插入：因为我们是头插，上一个插入的一定是当前链表的 head
                if *head == u32::MAX || self.node_word[*head as usize] != (i as u32) {
                    // M5：链表节点下标与 word 下标均由 usize 收敛为 u32，越界即报错
                    let new_node = u32_checked(self.node_word.len(), "倒排链表节点数")?;
                    self.node_word.push(u32_checked(i, "word 索引")?);
                    self.node_next.push(*head);
                    *head = new_node;
                }
            }
        }

        // 此刻数组再也不会增长了，把未使用的翻倍空间全部还给操作系统
        // self.flat_data.shrink_to_fit();
        // self.word_starts.shrink_to_fit();
        // self.word_lens.shrink_to_fit();
        // self.word_weights.shrink_to_fit();
        // self.node_word.shrink_to_fit();
        // self.node_next.shrink_to_fit();
        // self.pair_head.shrink_to_fit();
        Ok(())
    }

    /// A1 核心：升级 flat_data 从 Bytes(u8) 到 Wide(u32)。
    /// 一次性 memcpy，之后该 Shard 全程保持 Wide。
    fn ensure_wide(&mut self) {
        if let FlatData::Bytes(b) = &self.flat_data {
            let wide: Vec<u32> = b.iter().map(|&x| x as u32).collect();
            self.flat_data = FlatData::Wide(wide);
        }
    }

    fn apply_merge(
        &mut self,
        best_pair: u64,
        new_id: u32,
        deltas: &mut AHashMap<u64, i64>,
    ) -> Result<(), BbpeError> {
        let p0 = (best_pair >> 32) as u32;
        let p1 = (best_pair & 0xFFFFFFFF) as u32;

        // 【优化：从链表遍历收集受影响的词】
        let mut head = match self.pair_head.remove(&best_pair) {
            Some(h) => h,
            None => return Ok(()),
        };

        // 【精髓】：清空缓冲区并收集索引
        self.merge_buffer.clear();
        while head != u32::MAX {
            self.merge_buffer.push(self.node_word[head as usize]);
            head = self.node_next[head as usize];
        }

        // 【借用欺骗魔法】：用 std::mem::take 把 Vec 偷出来，骗过编译器！
        let word_indices = std::mem::take(&mut self.merge_buffer);

        // 【A1 核心】：new_id ≥ 256，必须升级到 Wide(u32) 才能写入
        self.ensure_wide();
        let flat = match &mut self.flat_data {
            FlatData::Wide(w) => w,
            FlatData::Bytes(_) => unreachable!("ensure_wide 后必为 Wide"),
        };

        for &idx in &word_indices {
            let start = self.word_starts[idx as usize] as usize;
            let mut len = self.word_lens[idx as usize] as usize;
            let weight = self.word_weights[idx as usize] as i64;

            // 1. 扣除旧词的所有连接贡献
            let slice_old = &flat[start..start + len];
            for w in slice_old.windows(2) {
                let pair = pack_pair(w[0], w[1]);
                record_change(deltas, pair, -weight);
            }

            // 2. 原地合并 p0, p1 -> new_id (双指针法，避免跨词内存移动)
            let mut write_p = start;
            let mut read_p = start;
            let end = start + len;
            while read_p < end {
                if read_p + 1 < end && flat[read_p] == p0 && flat[read_p + 1] == p1 {
                    flat[write_p] = new_id;
                    read_p += 2;
                } else {
                    flat[write_p] = flat[read_p];
                    read_p += 1;
                }
                write_p += 1;
            }

            // 更新当前词在原数组中的有效长度
            len = write_p - start;
            // M5：合并只会缩短词长，此处理论上不可能越界；仍显式检查以防未来改动
            self.word_lens[idx as usize] = u32_checked(len, "合并后 word 长度")?;

            // 3. 注入新词的连接贡献，并更新索引
            let slice_new = &flat[start..start + len];
            for w in slice_new.windows(2) {
                let pair = pack_pair(w[0], w[1]);
                record_change(deltas, pair, weight);

                // 【绝杀修复的数组链表版】
                if w[0] == new_id || w[1] == new_id {
                    let head = self.pair_head.entry(pair).or_insert(u32::MAX);
                    if *head == u32::MAX || self.node_word[*head as usize] != idx {
                        // M5：倒排链表节点下标越界检查
                        let new_node = u32_checked(self.node_word.len(), "倒排链表节点数")?;
                        self.node_word.push(idx);
                        self.node_next.push(*head);
                        *head = new_node;
                    }
                }
            }
        }

        self.merge_buffer = word_indices;
        Ok(())
    }

    /// 轻量级 Arena 垃圾回收器
    fn gc_arena(&mut self) -> Result<(), BbpeError> {
        // 1. 清空旧链表 (只清空数据，容量保持不变，0系统分配开销)
        self.pair_head.clear();
        self.node_word.clear();
        self.node_next.clear();

        // 2. 重新扫描当前存活的有效词汇，重建绝对紧凑的链表 (自动丢弃所有死节点)
        //    flat_data 此时可能为 Bytes 或 Wide，用宏消除两态重复
        macro_rules! rebuild {
            ($data:expr, $conv:expr) => {{
                let data = $data;
                let conv = $conv;
                for i in 0..self.word_starts.len() {
                    let start = self.word_starts[i] as usize;
                    let len = self.word_lens[i] as usize;
                    let slice = &data[start .. start + len];
                    for w in slice.windows(2) {
                        let pair = pack_pair(conv(w[0]), conv(w[1]));
                        let head = self.pair_head.entry(pair).or_insert(u32::MAX);
                        if *head == u32::MAX || self.node_word[*head as usize] != (i as u32) {
                            // M5：倒排链表下标越界检查
                            let new_node = u32_checked(self.node_word.len(), "倒排链表节点数")?;
                            self.node_word.push(u32_checked(i, "word 索引")?);
                            self.node_next.push(*head);
                            *head = new_node;
                        }
                    }
                }
                Ok::<(), BbpeError>(())
            }};
        }
        match &self.flat_data {
            FlatData::Bytes(b) => rebuild!(b, |x: u8| x as u32),
            FlatData::Wide(w) => rebuild!(w, |x: u32| x),
        }?;

        // ==========================================
        // 【A3】：物理逼退 —— 将多余的容量还给 OS
        // 文本在合并中越来越短，链表长度单调递减。gc_arena 重建后若容量利用率 < 50%，
        // 则缩容到当前长度的 1.25 倍 (留少量余量防止后续 merge 立即再次扩容)。
        // ==========================================
        let nw_len = self.node_word.len();
        let nw_cap = self.node_word.capacity();
        if nw_cap > 100_000 && nw_cap > nw_len * 2 {
            let target = nw_len + nw_len / 4;
            self.node_word.shrink_to(target);
            self.node_next.shrink_to(target);
        }
        let ph_len = self.pair_head.len();
        let ph_cap = self.pair_head.capacity();
        if ph_cap > 50_000 && ph_cap > ph_len * 2 {
            self.pair_head.shrink_to(ph_len + ph_len / 4);
        }
        Ok(())
    }

    /// B3: 深度 GC + 渐进冻结
    /// 判定每个 word：若其所有 pair 的全局频次 < m (即不在 active_pairs 中)，
    /// 则冻结(从 flat_data/索引移除字节内容)。紧凑化 flat_data，只保留存活 word。
    /// 等价性：冻结 word 的 pair 全程 < m，永不被选中，其 global_counts 贡献为常数保留。
    fn gc_freeze(
        &mut self,
        global_counts: &AHashMap<u64, u64>,
        m: u64,
    ) -> Result<usize, BbpeError> {
        // 1+2: 判定存活 + 紧凑化 flat_data/starts/lens/weights
        let (new_data, new_starts, new_lens, new_weights) = match &self.flat_data {
            FlatData::Bytes(b) => {
                let (d, s, l, w) = compact_freeze(
                    b,
                    &self.word_starts,
                    &self.word_lens,
                    &self.word_weights,
                    global_counts,
                    m,
                    |x: u8| x as u32,
                )?;
                (FlatData::Bytes(d), s, l, w)
            }
            FlatData::Wide(w) => {
                let (d, s, l, w) = compact_freeze(
                    w,
                    &self.word_starts,
                    &self.word_lens,
                    &self.word_weights,
                    global_counts,
                    m,
                    |x: u32| x,
                )?;
                (FlatData::Wide(d), s, l, w)
            }
        };
        let frozen_n = self.word_starts.len().saturating_sub(new_starts.len());
        self.flat_data = new_data;
        self.word_starts = new_starts;
        self.word_lens = new_lens;
        self.word_weights = new_weights;

        // 3: 重建倒排索引 (只存活 word)
        self.pair_head.clear();
        self.node_word.clear();
        self.node_next.clear();
        macro_rules! rebuild_idx {
            ($data:expr, $conv:expr) => {{
                let data = $data;
                let conv = $conv;
                for i in 0..self.word_starts.len() {
                    let start = self.word_starts[i] as usize;
                    let len = self.word_lens[i] as usize;
                    let slice = &data[start .. start + len];
                    for w in slice.windows(2) {
                        let pair = pack_pair(conv(w[0]), conv(w[1]));
                        let head = self.pair_head.entry(pair).or_insert(u32::MAX);
                        if *head == u32::MAX || self.node_word[*head as usize] != (i as u32) {
                            // M5：倒排链表下标越界检查
                            let new_node = u32_checked(self.node_word.len(), "倒排链表节点数")?;
                            self.node_word.push(u32_checked(i, "word 索引")?);
                            self.node_next.push(*head);
                            *head = new_node;
                        }
                    }
                }
                Ok::<(), BbpeError>(())
            }};
        }
        match &self.flat_data {
            FlatData::Bytes(b) => rebuild_idx!(b, |x: u8| x as u32),
            FlatData::Wide(w) => rebuild_idx!(w, |x: u32| x),
        }?;

        // 4: 物理缩容 (复用 A3 水位线)
        let nw_len = self.node_word.len();
        let nw_cap = self.node_word.capacity();
        if nw_cap > 100_000 && nw_cap > nw_len * 2 {
            let target = nw_len + nw_len / 4;
            self.node_word.shrink_to(target);
            self.node_next.shrink_to(target);
        }
        let ph_len = self.pair_head.len();
        let ph_cap = self.pair_head.capacity();
        if ph_cap > 50_000 && ph_cap > ph_len * 2 {
            self.pair_head.shrink_to(ph_len + ph_len / 4);
        }
        Ok(frozen_n)
    }

    /// 智能触发器
    fn maybe_gc(&mut self) -> Result<(), BbpeError> {
        // 【优化红线】：当长度达到容量的 85% 时就触发紧凑化覆写
        // 绝对阻止 Vec 向操作系统申请双倍内存扩容！
        if self.node_word.capacity() > 100_000
            && self.node_word.len() > self.node_word.capacity() * 85 / 100
        {
            self.gc_arena()?;
        }
        Ok(())
    }
}

// =========================================================
// 系统运行配置 (Runtime & System)
// =========================================================
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub batch_size: usize,
    pub num_shards: usize,
    pub parallel_index_build: bool,
    pub max_chunk_bytes: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            batch_size: 50_000,
            num_shards: rayon::current_num_threads(),
            parallel_index_build: true,
            max_chunk_bytes: 512,
        }
    }
}

// =========================================================
// 算法训练配置 (Trainer Domain)
// =========================================================
#[derive(Clone, Debug)]
pub struct TrainerConfig {
    pub vocab_size: usize,
    pub min_frequency: u32,
    pub special_tokens: Vec<String>,
    /// 预分词正则 pattern。【唯一来源】未显式设置时由 `new` 填入 [`crate::pre_tokenizer::DEFAULT_PATTERN`]。
    pub pre_tokenizer_pattern: String,
}

impl TrainerConfig {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            min_frequency: 2,
            special_tokens: Vec::new(),
            pre_tokenizer_pattern: crate::pre_tokenizer::DEFAULT_PATTERN.to_string(),
        }
    }

    /// 链式 setter：自定义 pre-tokenizer 正则
    pub fn with_pre_tokenizer_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pre_tokenizer_pattern = pattern.into();
        self
    }
}

// =========================================================
// BBPETrainer 核心引擎改造
// =========================================================

/// R4-20 / R5-3：special token 列表的**唯一**校验实现。
///
/// 空串会让 encode 的 AhoCorasick 空 pattern 在每个字节位匹配（含 UTF-8
/// 字符内部）⟹ 非字符边界切片 panic；重复项在 added_tokens 里互相覆盖，
/// id 布局静默漂移。`new()` 与 `train_inner()` 共用本函数：`trainer_config`
/// 是 pub 字段，调用方可在构造**之后**改写 special_tokens 绕过构造期校验
/// （R5-3），训练前的复查把“构造合法、训练时非法”的割裂堆死在入口。
fn validate_special_tokens(tokens: &[String]) -> Result<(), BbpeError> {
    if let Some(t) = tokens.iter().find(|t| t.is_empty()) {
        return Err(BbpeError::InvalidSpecialTokens(format!(
            "special token 列表含空字符串（{t:?}）：空 token 无语义且会使 encode panic"
        )));
    }
    if tokens.len()
        != tokens
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    {
        return Err(BbpeError::InvalidSpecialTokens(
            "special token 列表含重复项：重复项在 added_tokens 里互相覆盖，id 布局静默漂移"
                .to_string(),
        ));
    }
    Ok(())
}

pub struct BBPETrainer {
    pre_tokenizer: PreTokenizer,
    pub state: BpeState,
    pub trainer_config: TrainerConfig,
    pub runtime_config: RuntimeConfig,
}

// R6-1 P1：训练态要整体 move 进 worker 线程，必须在类型层面证明 Send。
// 与 `tokenizer.rs` 的 BBPETokenizer 一样用编译期静态断言，零 unsafe；
// 若将来给 trainer 加了 !Send 字段，这里会立即编译失败而不是运行时炸。
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<BBPETrainer>();
};

impl BBPETrainer {
    pub fn new(
        trainer_config: TrainerConfig,
        runtime_config: Option<RuntimeConfig>,
    ) -> Result<Self, BbpeError> {
        let runtime_config = runtime_config.unwrap_or_default();

        // R4-8：num_shards = 0 会在活跃词分片（`assign_shard` 的 `% n`）处
        // **除零 panic**。Python 侧 `RuntimeConfig().with_num_shards(0)` 可直达，
        // 故在构造期显式拒绝，把 panic 变成可操作的参数错误。
        if runtime_config.num_shards == 0 {
            return Err(BbpeError::InvalidRuntimeConfig(
                "num_shards 必须 ≥ 1（0 会在活跃词分片取模时除零 panic）".to_string(),
            ));
        }
        if runtime_config.batch_size == 0 {
            return Err(BbpeError::InvalidRuntimeConfig(
                "batch_size 必须 ≥ 1（0 会让批次熔断永远触发，逐行处理）".to_string(),
            ));
        }

        // R4-20：空 / 重复 special token 必须在**训练端**拒绝。
        // 加载端（BpeState::load 解析期 + V6）此前就会拒绝，于是出现
        // "训练保存成功、加载即失败"的割裂；空串还会让 encode 的
        // AhoCorasick 空 pattern 在每个字节位匹配（含 UTF-8 字符内部）⟹
        // 非字符边界切片 panic。校验逻辑与 train_inner 的复查共用同一实现。
        validate_special_tokens(&trainer_config.special_tokens)?;

        // 词表必须同时装下 256 个基础字节与全部 special tokens（尾部坑位）。
        // 若 vocab_size ≤ 256 + n_special，base_vocab_size < 256，special id 会
        // **回卷覆盖基础字节条目**（V1 的字节完整性被静默破坏，规范 §6.4）。
        let min_vocab_size = 256 + trainer_config.special_tokens.len();
        if trainer_config.vocab_size <= min_vocab_size {
            return Err(BbpeError::VocabSizeTooSmall(min_vocab_size));
        }

        let mut vocab = HashMap::new();
        for i in 0..=255 {
            vocab.insert(i, vec![i as u8]);
        }

        // 【核心修复】：直接在这里注入特殊字符，彻底干掉 Python 侧的二次拼装！
        let mut special_tokens = HashMap::new();
        let base_vocab_size = trainer_config
            .vocab_size
            .saturating_sub(trainer_config.special_tokens.len());

        for (i, token) in trainer_config.special_tokens.iter().enumerate() {
            let id = (base_vocab_size + i) as i32;
            special_tokens.insert(token.clone(), id);
            // 同步塞入基础词表，防止查找断层
            vocab.insert(id, token.as_bytes().to_vec());
        }

        let state = BpeState {
            vocab,
            merges: HashMap::new(),
            merges_fast: rustc_hash::FxHashMap::default(),
            special_tokens, // <--- 现在它拥有了完整的特殊字符！
            pre_tokenizer_pattern: trainer_config.pre_tokenizer_pattern.clone(),
            meta: None, // 由 train() 在结束时填充 train 段（模型契约 v1 §7.1）
            // A4：新建 state 尚未落盘，不声明任何 format（与 format_version 一致）
            format: None,
            format_version: None,
            // m8：新建的 state 由调用方显式指定 pattern，不存在"回退"
            legacy_pattern_fallback: false,
        };

        Ok(Self {
            // m4：把 RuntimeConfig::max_chunk_bytes 真正接到预分词器上，
            // 使 `meta.train.max_chunk_bytes` 记录的值 == 实际切分行为。
            pre_tokenizer: PreTokenizer::with_max_chunk_bytes(
                &trainer_config.pre_tokenizer_pattern,
                runtime_config.max_chunk_bytes,
            )?,
            state,
            trainer_config,
            runtime_config,
        })
    }

    /// 替换实际预分词器（R4-23：同步 state / config / meta 三处记录）。
    ///
    /// 历史裁定（B3）选择"只换实际预分词器"，但结果是：`state.pre_tokenizer_pattern`
    /// 与 `trainer_config.pre_tokenizer_pattern` 仍记着旧 pattern，保存出的模型
    /// meta.train 与实际切分行为不符（契约记录 ≠ 事实），且再次 `reset()` 会用
    /// 旧 pattern 重建 state。现在把 pattern 与 chunk 上限一并同步。
    pub fn set_pre_tokenizer(&mut self, pre_tokenizer: PreTokenizer) {
        self.state.pre_tokenizer_pattern = pre_tokenizer.pattern().to_string();
        self.trainer_config.pre_tokenizer_pattern = pre_tokenizer.pattern().to_string();
        // m4/R4-23：meta.train.max_chunk_bytes 由 runtime_config 写出，必须同步
        self.runtime_config.max_chunk_bytes = pre_tokenizer.max_chunk_bytes();
        self.pre_tokenizer = pre_tokenizer;
    }

    /// 重置为"未训练"状态，使其可再次用于训练（见 [`Self::train`] 的幂等性说明）。
    ///
    /// A1：训练是**有状态**的 —— 产出的 merges / vocab 全部累积在本实例的
    /// [`Self::state`] 里。`new_id = 256 + i` 每轮从 256 重启，直接二次训练会与
    /// 残留 id 撞车，累积后 `merges` 的 new_id 集合不再连续（V3 违反）。因此
    /// 未显式 reset 时，二次 `train` 会直接报错而非静默产出损坏模型。
    ///
    /// 本方法按 `new()` 的同一规则重建词表（256 基础字节 + special tokens 尾部坑位），
    /// 并清空 `merges` / `meta`，使 `reset` 后的实例与全新 `new` 的实例等价。
    pub fn reset(&mut self) {
        let mut vocab = HashMap::new();
        for i in 0..=255 {
            vocab.insert(i, vec![i as u8]);
        }

        // 与 new() 同一布局：special token 占据词表尾部
        let base_vocab_size = self
            .trainer_config
            .vocab_size
            .saturating_sub(self.trainer_config.special_tokens.len());
        let mut special_tokens = HashMap::new();
        for (i, token) in self.trainer_config.special_tokens.iter().enumerate() {
            let id = (base_vocab_size + i) as i32;
            special_tokens.insert(token.clone(), id);
            vocab.insert(id, token.as_bytes().to_vec());
        }

        self.state = BpeState {
            vocab,
            merges: HashMap::new(),
            merges_fast: rustc_hash::FxHashMap::default(),
            special_tokens,
            pre_tokenizer_pattern: self.trainer_config.pre_tokenizer_pattern.clone(),
            meta: None,
            format: None,
            format_version: None,
            legacy_pattern_fallback: false,
        };
    }

    /// 当前是否已训练过（`merges` 非空）。供调用方在复用 trainer 前自查。
    pub fn is_trained(&self) -> bool {
        !self.state.merges.is_empty()
    }

    /// 训练。与 [`Self::train_with_stats`] 的差别仅为**不采集输入侧计数**。
    ///
    /// **A1 幂等性**：本方法**不是**幂等的 —— 重复调用会返回
    /// [`BbpeError::AlreadyTrained`]，而不是静默产出违反 V3 的损坏模型。
    /// 需要重训请先 [`Self::reset`]，或新建一个 trainer。
    pub fn train(&mut self, reader: &CorpusReader) -> Result<(), BbpeError> {
        self.train_inner(reader, None, None)
    }

    /// 带输入侧计数上报的训练（规范 §6.6）：`skipped_lines` / `replacement_chars`
    /// 等会被写入 `meta.corpus`，让静默行为可见（G4）。
    pub fn train_with_stats(
        &mut self,
        reader: &CorpusReader,
        stats: Arc<CorpusStats>,
    ) -> Result<(), BbpeError> {
        self.train_inner(reader, Some(stats), None)
    }

    /// R6-1：带取消令牌的训练。
    ///
    /// `cancel = None` 时与 [`Self::train_with_stats`] 逐位一致（检查点只多一次
    /// 原子读）。取消命中时返回 [`BbpeError::Cancelled`]：训练态留在本实例里但
    /// **不写 `meta.train`、不算完成**；调用方必须拒绝 `save()`，只能 `reset()`
    /// 后重训或新建实例。
    pub fn train_with_cancel(
        &mut self,
        reader: &CorpusReader,
        stats: Option<Arc<CorpusStats>>,
        cancel: Option<&CancelToken>,
    ) -> Result<(), BbpeError> {
        self.train_inner(reader, stats, cancel)
    }

    fn train_inner(
        &mut self,
        reader: &CorpusReader,
        stats: Option<Arc<CorpusStats>>,
        cancel: Option<&CancelToken>,
    ) -> Result<(), BbpeError> {
        // ---- R6-1：取消检查点基础设施 ----
        // 进度条用 RAII 收尾：任何 Err/取消/普通错误路径都 abandon，只有走到
        // 末尾标记 finished 才显示"训练完成"。过去 `?` 提前返回会把进度条留在
        // 屏幕上，Cancel 路径更不能显示完成。
        struct PbGuard<'a> {
            pb: &'a ProgressBar,
            finished: bool,
        }
        impl Drop for PbGuard<'_> {
            fn drop(&mut self) {
                if !self.finished {
                    self.pb.abandon_with_message("训练已中止");
                }
            }
        }

        let cancel_err = |done: usize| {
            // R6-1 P1：主线程不 join worker，靠令牌里的进度位取回 provenance。
            if let Some(c) = cancel {
                c.set_progress(done);
            }
            BbpeError::Cancelled {
                merges_done: done,
                reason: cancel.map(|c| c.reason()).unwrap_or(CancelReason::User),
            }
        };

        // 粗粒度检查点：poll() 会按间隔探测信号，命中即 return Err。
        // 细粒度检查点（热循环 / rayon 闭包）只读标志，不触发探测。
        // 两者都必须是 `return Err` —— train_inner 末尾无条件写 meta.train
        // 并 Ok(())，break 会产出"训练完成"的半截模型。
        macro_rules! ck {
            ($done:expr) => {
                if let Some(c) = cancel {
                    if c.poll() {
                        return Err(cancel_err($done));
                    }
                }
            };
        }
        macro_rules! ck_fine {
            ($done:expr) => {
                if let Some(c) = cancel {
                    if c.is_cancelled() {
                        return Err(cancel_err($done));
                    }
                }
            };
        }

        if cancel.is_some() {
            ck!(0);
        }

        // ---- A1：幂等性守卫 ----
        // 训练会把 merges / vocab **累积**进本实例的 state，而 `new_id = 256 + i`
        // 每轮从 256 重启。若允许二次训练：新写出的 new_id 会与上一轮残留的 id
        // 撞车，且两轮的 new_id 交错后不再构成连续区间 {256, …} ⟹ 违反 V3。
        // 更糟的是 trainer.rs 内 `self.state.vocab[&id]` 是直接索引，id 被覆盖后
        // 会读到错误字节（甚至 panic）。实测：vocab=400 先训 A（32 merges）再训 B，
        // merges 变成 54 且 `validate()` 报「V3 违反：new_id 集合不是连续区间」。
        // 因此这里显式失败，把"静默产出损坏模型"变成可操作的报错。
        if !self.state.merges.is_empty() {
            return Err(BbpeError::AlreadyTrained(self.state.merges.len()));
        }

        // R5-3：`trainer_config` 是 pub 字段，调用方可在 `new()` 之后直接改写
        // `special_tokens`（绕过构造期校验）；`reset()` 还会按改写后的配置重建
        // state。这里用与 `new()` 同一份实现复查，非法配置在训练入口即失败，
        // 不会产出“保存成功、加载即崩”的模型。Python 侧不受影响（配置在构造
        // 前设置），该防线针对的是 Rust 公开 API 的边界路径。
        validate_special_tokens(&self.trainer_config.special_tokens)?;

        let min_vocab_size = 256 + self.trainer_config.special_tokens.len();
        if self.trainer_config.vocab_size <= min_vocab_size {
            return Err(BbpeError::VocabSizeTooSmall(min_vocab_size));
        }
        if self.trainer_config.vocab_size > 200_000 {
            // 上界（规范 §6.4）：不报错，但 > 200,000 时推论 7 未重验，
            // merges 序列可能开始受 min_frequency 影响
            tracing::warn!(
                "vocab_size = {} 超过 200,000：该规模下推论 7 未重验（规范 §6.4），\
                 merges 结果可能受 min_frequency 影响",
                self.trainer_config.vocab_size
            );
        }

        // 获取运行参数
        let batch_size = self.runtime_config.batch_size;

        // R4-8：`runtime_config` 是 pub 字段，构造后仍可能被改写为 0
        // （`new` 已校验一次，这里是第二道防线）⟹ 取模除零 panic 变成参数错误。
        if self.runtime_config.num_shards == 0 {
            return Err(BbpeError::InvalidRuntimeConfig(
                "num_shards 必须 ≥ 1（0 会在活跃词分片取模时除零 panic）".to_string(),
            ));
        }

        info!("阶段 1: 启动分片哈希架构并行读取与预处理...");
        let bench_t0 = std::time::Instant::now();
        let mut bench_t_phase = std::time::Instant::now();

        let shards_count = self.runtime_config.num_shards;

        // ==========================================
        // 内存分片哈希架构：256 个无锁全局哈希分片统计 word 频次
        // ==========================================
        const NUM_SHARDS: usize = 256;
        let global_shards: Vec<std::sync::Mutex<AHashMap<Vec<u8>, u64>>> = (0..NUM_SHARDS)
            .map(|_| std::sync::Mutex::new(AHashMap::with_capacity(20_000)))
            .collect();

        // A7：拆成"算哈希"与"映射到区间"两步。
        //
        // 原实现里 `shard_idx` 已把哈希截断到低 8 位（`& (NUM_SHARDS-1)`，因为
        // global_shards 恰好 256 个），而 `assign_shard` 又在这**已截断**的值上
        // 取 `% n` —— 当 n 不整除 256 时，只有 `[0, 256 mod n)` 号分片会多拿到
        // 一份，分布天然不均。
        #[inline(always)]
        fn fnv1a64(bytes: &[u8]) -> u64 {
            let mut hash: u64 = 0xcbf29ce484222325;
            for &b in bytes {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash
        }

        /// 256 路分片：2 的幂次，用位与即可（此处截断是正确的）
        #[inline(always)]
        fn shard_idx(bytes: &[u8]) -> usize {
            (fnv1a64(bytes) as usize) & (NUM_SHARDS - 1)
        }

        /// shard 分配：把**完整 64 位**哈希映射到 `[0, n)`。
        ///
        /// A7：原写法 `shard_idx(bytes) % n` 在低 8 位上取模，非 2 的幂次下
        /// 负载偏斜。实测偏斜幅度（6 万个 `word{i}` 词，最大分片 vs 平均）：
        /// 修复前 n=12 偏斜 +4.1%、n=24 偏斜 +4.6%；而 `num_shards` 默认取
        /// `rayon::current_num_threads()`，12 核 / 24 核是常见工作站配置。
        ///
        /// 方案选择（实测 6 万词 × 3 类输入 × n∈{2,6,8,10,12,16,24,32,64}）：
        /// 单纯"乘法散列 `(h * n) >> 64`"**不可用** —— 它取哈希的**高位**，
        /// 而 FNV-1a 的高位雪崩很差，对 `word0/word1/...` 这类顺序输入
        /// 反而更差（实测 n=64 偏斜 +79.2%/-68.0%，比修复前更糟）。
        /// 加 splitmix64 finalizer 能救回但仍不及原方案。
        /// 最终采用**完整 64 位哈希直接取模**：对非 2 的幂次 n，它用到哈希的
        /// 所有位（而非只有低 8 位），实测在顺序输入集上全面优于修复前
        /// （n=12: +4.1% → +0.5%；n=24: +4.6% → +0.6%），其余输入集持平。
        ///
        /// 附带说明：这**不改变训练输出**（merges 序列）。分片归属只影响负载均衡
        /// —— `apply_merge` 产生的 delta 按 shard 汇总后是同一组整数求和，与顺序
        /// 无关；`global_counts` 的最终值与 `pq` 的选择因此不受影响。
        #[inline(always)]
        fn assign_shard(bytes: &[u8], n: usize) -> usize {
            (fnv1a64(bytes) % n as u64) as usize
        }

        let mut lines_batch = Vec::with_capacity(batch_size);
        let mut current_batch_bytes = 0;
        let max_batch_bytes = 64 * 1024 * 1024; // 64 MB 物理熔断

        // A2：预分词错误必须让训练**硬失败**，不能只打警告。
        // `par_iter().for_each` 无法直接 `?`，故用 try_for_each + 原子标志
        // 记住首个错误（多条行同时失败时报告第一条最有诊断价值）。
        let process_batch = |batch: &[String]| -> Result<(), BbpeError> {
            let pretokenize_err: std::sync::Mutex<Option<BbpeError>> = std::sync::Mutex::new(None);
            batch.par_iter().try_for_each(|line| {
                self.pre_tokenizer
                    .tokenize_with_checked(line, |bytes_chunk, _offset| {
                        let idx = shard_idx(bytes_chunk);
                        let mut map = global_shards[idx].lock().unwrap();

                        // 借用查询魔法：99.9% 命中缓存，0 次堆分配
                        if let Some(count) = map.get_mut(bytes_chunk) {
                            *count += 1;
                        } else {
                            map.insert(bytes_chunk.to_vec(), 1);
                        }
                    })
                    .inspect_err(|e| {
                        // 只记录首个错误：后续错误多半是同一 pattern 的同一问题
                        let mut slot = pretokenize_err.lock().unwrap();
                        if slot.is_none() {
                            *slot = Some(crate::error::BbpeError::PreTokenize(format!(
                                "语料预分词失败（该行**未被计入**词频，且此行不可恢复）: {e}"
                            )));
                        }
                    })
            })?;
            // 取回首个错误并向上传播 ⟹ 训练终止，绝不静默丢语料
            match pretokenize_err.into_inner().unwrap() {
                Some(e) => Err(e),
                None => Ok(()),
            }
        };

        // 循环引擎：双重熔断驱动
        // R4-7：改用**严格**迭代器 —— IO 错误（NFS/坏块/句柄错误）不再被当作
        // EOF，否则后续语料静默消失、训练照常保存"截断模型"并返回成功。
        let lines: Box<dyn Iterator<Item = Result<String, BbpeError>>> = match &stats {
            Some(s) => reader.iter_lines_checked(Some(s.clone()))?,
            None => reader.iter_lines_checked(None)?,
        };
        // R6-1：阶段 1 的检查点。每 1024 行一次粗检 + 每个批次边界一次，
        // 避免 64MB / 50k 行的批次熔断成为不可中断窗口。
        let mut lines_seen: usize = 0;
        for line in lines {
            lines_seen += 1;
            if lines_seen % 1024 == 0 {
                ck!(0);
            }
            let line = line?; // IO 错误 ⟹ 训练显式失败（不保存截断模型）
            current_batch_bytes += line.len();
            lines_batch.push(line);

            if current_batch_bytes >= max_batch_bytes || lines_batch.len() >= batch_size {
                ck!(0);
                process_batch(&lines_batch)?; // A2：预分词失败 ⟹ 训练终止
                lines_batch.clear();
                current_batch_bytes = 0;
            }
        }
        ck!(0);
        if !lines_batch.is_empty() {
            process_batch(&lines_batch)?; // A2：同上
        }

        // G4：**0 行可用**绝不可能是用户意图（同 A2「未覆盖间隙是硬错误」口径）。
        //
        // 此前该情形会一路跑到结束并返回 Ok：产出"256 个基础字节 + 0 条 merges"
        // 的空模型，`inspect --validate-only` 也判它合法（V1–V7 对空 merges 成立），
        // 用户只看到一句 ⚠️ 告警和一个"成功保存"的模型文件。
        // 最典型的触发：把纯文本按默认 `--format jsonl` 读入 ⟹ 每行都解析失败。
        if lines_seen == 0 {
            // 成因与处置按**实际情形**分支。把「空文件」「只有空行」「格式不符」混成
            // 一句话，会把用户引到错误的修法上（去改 --format，而文件本来就是空的），
            // 这比不报错更浪费排查时间。
            let (cause, remedy) = match &stats {
                Some(s) => {
                    let (total, _, skipped, _) = s.snapshot();
                    if total == 0 {
                        (
                            "输入文件为空（0 行 / 0 字节）".to_string(),
                            "提供至少一行数据。若确实需要一个「256 个基础字节 + 0 merges」\
                             的占位模型，请直接复制一个已有模型 —— 训练一个与输入无关的\
                             产物不是有效调用"
                                .to_string(),
                        )
                    } else if skipped == 0 {
                        (
                            format!("物理行 {total} 行，但没有任何一行含可训练内容（全部为空行）"),
                            "提供至少一行非空文本；空行会被 reader 过滤，不参与训练".to_string(),
                        )
                    } else if reader.kind().starts_with("jsonl") {
                        (
                            format!("物理行 {total} 行，其中 {skipped} 行全部解析失败（0 行可用）"),
                            "读到的内容与 jsonl 期望不符。纯文本请显式指定 text 格式 —— \
                             CLI：`--format text`；Python：用 `TextFileReader` 而不是 \
                             `JsonlFileReader`"
                                .to_string(),
                        )
                    } else {
                        (
                            format!("物理行 {total} 行，其中 {skipped} 行全部不可用（0 行可用）"),
                            "检查文件编码 / 行尾与 reader 的语义是否匹配".to_string(),
                        )
                    }
                }
                None => (
                    "未读到任何可用行".to_string(),
                    "确认输入文件非空，且内容与所用 reader 的格式匹配".to_string(),
                ),
            };
            return Err(BbpeError::EmptyCorpus(format!(
                "{}（reader={}）{cause}。\n处置：{remedy}。",
                reader.filepath(),
                reader.kind(),
            )));
        }

        // 平滑回收分片，拼装成完整的 word_freqs
        let total_unique_words: usize = global_shards.iter().map(|s| s.lock().unwrap().len()).sum();
        let mut word_freqs: AHashMap<Vec<u8>, u64> = AHashMap::with_capacity(total_unique_words);
        let mut merged_words: usize = 0;
        for shard in global_shards {
            let map = shard.into_inner().unwrap();
            for (k, v) in map {
                merged_words += 1;
                if merged_words % 65536 == 0 {
                    ck!(0);
                }
                word_freqs.insert(k, v);
            }
        }
        word_freqs.shrink_to_fit();

        info!(
            "[bench] 阶段1(读取预处理) 耗时: {:.3}s",
            bench_t_phase.elapsed().as_secs_f64()
        );
        info!("阶段 2: 初始化 {} 个并行分片 (Shards)...", shards_count);
        bench_t_phase = std::time::Instant::now();

        let mut shards = vec![Shard::new(); shards_count];

        // ==========================================
        // 【B2】：pair_freq 计算 + 冻结判定 (内存 word_freqs 两趟扫描)
        // 第一趟：算初始 pair 频次 (含全部 word 贡献)
        // 第二趟：冻结判定 + 统计 shard 容量 (不物化 sorted_freqs)
        // 等价性：word_freqs 遍历 = 内存 HashMap 统计，仅顺序变化，不影响结果。
        // ==========================================
        let m = self.trainer_config.min_frequency as u64;
        let mut pair_freq: AHashMap<u64, u64> = AHashMap::default();
        let mut total_words: usize = 0;

        // 第一趟：算 pair_freq
        for (bytes, &count) in &word_freqs {
            total_words += 1;
            if total_words % 4096 == 0 {
                ck!(0);
            }
            for w in bytes.windows(2) {
                *pair_freq
                    .entry(pack_pair(w[0] as u32, w[1] as u32))
                    .or_insert(0) += count;
            }
        }

        // 第二趟：冻结判定 + 统计 shard 容量
        let mut active_words: usize = 0;
        let mut shard_byte_sizes = vec![0usize; shards_count];
        let mut shard_chunk_counts = vec![0usize; shards_count];

        let mut scanned2: usize = 0;
        for (bytes, _) in &word_freqs {
            scanned2 += 1;
            if scanned2 % 4096 == 0 {
                ck!(0);
            }
            let active = bytes.windows(2).any(|w| {
                let pair = pack_pair(w[0] as u32, w[1] as u32);
                pair_freq.get(&pair).copied().unwrap_or(0) >= m
            });
            if active {
                active_words += 1;
                // 用 FNV hash 做确定性 shard 分配 (替代排序+贪心，等价但无需物化)
                let idx = assign_shard(bytes, shards_count);
                shard_byte_sizes[idx] += bytes.len();
                shard_chunk_counts[idx] += 1;
            }
        }
        let frozen_words = total_words.saturating_sub(active_words);
        info!(
            "B2: 总 {} unique words, 活跃 {}, 冻结 {} (m={})",
            total_words, active_words, frozen_words, m
        );

        // 一次性精准预分配 shard 容量
        for i in 0..shards_count {
            shards[i].flat_data.reserve_exact(shard_byte_sizes[i]);
            shards[i].word_starts.reserve_exact(shard_chunk_counts[i]);
            shards[i].word_lens.reserve_exact(shard_chunk_counts[i]);
            shards[i].word_weights.reserve_exact(shard_chunk_counts[i]);
        }

        // 第三趟：物理装载活跃 word (不物化中间结构，直接写入 shard)
        let mut scanned3: usize = 0;
        for (bytes, &count) in &word_freqs {
            scanned3 += 1;
            if scanned3 % 4096 == 0 {
                ck!(0);
            }
            let active = bytes.windows(2).any(|w| {
                let pair = pack_pair(w[0] as u32, w[1] as u32);
                pair_freq.get(&pair).copied().unwrap_or(0) >= m
            });
            if active {
                let idx = assign_shard(bytes, shards_count);
                // M5：容量超限立即失败，不再静默截断
                shards[idx].add_chunk(bytes, count)?;
            }
        }

        // word_freqs 已全部装载到 shard，释放内存
        drop(word_freqs);

        // 阶段 2 末尾：并行构建倒排索引
        info!(
            "构建倒排索引 (并行模式: {})...",
            self.runtime_config.parallel_index_build
        );
        ck!(0);
        if self.runtime_config.parallel_index_build {
            shards
                .par_iter_mut()
                .try_for_each(|shard| shard.build_index())?;
        } else {
            shards
                .iter_mut()
                .try_for_each(|shard| shard.build_index())?;
        }
        ck!(0);

        // 精准计算合并次数 (极其关键！)
        // 目标合并次数 = 目标词表总数 - 256个基础字节 - 特殊字符占据的坑位
        let num_merges = self
            .trainer_config
            .vocab_size
            .saturating_sub(256)
            .saturating_sub(self.trainer_config.special_tokens.len());
        info!(
            "[bench] 阶段2(装载+索引) 耗时: {:.3}s",
            bench_t_phase.elapsed().as_secs_f64()
        );
        info!(
            "阶段 3: 并行 MapReduce 启动 (目标 {} 次 merges)",
            num_merges
        );
        bench_t_phase = std::time::Instant::now();

        let pb = ProgressBar::new(num_merges as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap(),
        );
        // R6-1：从此处起，任何 Err 路径都自动 abandon 进度条。
        let mut pb_guard = PbGuard {
            pb: &pb,
            finished: false,
        };

        // 【B2】：global_counts 直接复用 pair_freq (含活跃+冻结全部贡献，等价于基准全量统计)。
        // 冻结 word 的 pair 频次 < m，全程不被选中，其贡献作为常数驻留，无需动态更新。
        let mut global_counts: AHashMap<u64, u64> = pair_freq;

        // 【Step 2】：不将全部 590M pairs 填入 BinaryHeap (省 ~9.4GB)。
        // 只取 top-K 放入堆。由单调性引理：已有 pair 频次只减不增，
        // 新增 pair (含 new_id) 的 deltas 会被 push 进堆，故 top-K 覆盖所有候选。
        const HEAP_TOP_K: usize = 2_000_000; // 2M × 16B = 32MB
        // 2026-11-06 实测：阈值 10M 时，1GB/150k merges 触发 32 次全表重建，
        // 累计 96s（占 merge 阶段 24%，每次约 3s，因为要扫全部 global_counts
        // 并重建 top-K）。抬到 40M 后重建次数降到 ~1/4，代价是多占约 480MB 堆
        // 内存（40M × 16B − 10M × 16B），在 85GB 级训练里可忽略。
        const HEAP_GC_THRESHOLD: usize = 40_000_000; // 40M × 16B = 640MB

        fn refill_heap(
            global_counts: &AHashMap<u64, u64>,
            pq: &mut BinaryHeap<HeapEntry>,
            k: usize,
            cancel: Option<&CancelToken>,
            done: usize,
        ) -> Result<(), BbpeError> {
            // 用 min-heap 维护 top-K，O(n) 时间 O(k) 空间
            // R6-1：单次全表重建实测约 3s，必须能中途退出。
            use std::cmp::Reverse;
            let mut min_heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(k + 1);
            let mut scanned: usize = 0;
            for (&pair, &count) in global_counts {
                scanned += 1;
                if scanned % 65536 == 0 {
                    if let Some(c) = cancel {
                        if c.is_cancelled() {
                            c.set_progress(done);
                            return Err(BbpeError::Cancelled {
                                merges_done: done,
                                reason: c.reason(),
                            });
                        }
                    }
                }
                if count == 0 {
                    continue;
                }
                min_heap.push(Reverse(HeapEntry { count, pair }));
                if min_heap.len() > k {
                    min_heap.pop();
                }
            }
            *pq = min_heap.into_iter().map(|r| r.0).collect();
            Ok(())
        }

        let mut pq = BinaryHeap::new();
        refill_heap(&global_counts, &mut pq, HEAP_TOP_K, cancel, 0)?;

        let mut shard_active = vec![false; shards_count];
        // Delta 类型升级为 i64
        let mut thread_deltas: Vec<AHashMap<u64, i64>> = vec![AHashMap::default(); shards_count];

        // 模型契约 v1：记录训练终止原因与实际 merges 数（G1 可追溯 / 推论 7 判据）
        let mut actual_merges = num_merges;
        let mut end_reason = "vocab_size_reached";

        // ---------- 分阶段计时（诊断，默认不输出）----------
        // 大语料实测：150k merges 耗时约 2.6h，平均 ~62ms/merge。相对这个量级，
        // 每轮多 5 次 Instant::now()（~100ns）的开销可以忽略，换来的是"钱花在哪"。
        let mut t_select = 0f64; // 取最优 pair（堆 pop + 可能的 refill）
        let mut t_apply = 0f64; // 并行 apply_merge
        let mut t_inject = 0f64; // deltas 注入 global_counts + 堆更新
        let mut t_gc = 0f64; // 周期性冻结 / 容量水位保护
        let mut n_delta = 0u64; // deltas 总条数
        let mut n_push = 0u64; // 堆 push 次数
        let mut n_push_below_m = 0u64; // 其中频次 < m 的（永远不可能被选中）
        let mut n_remove = 0u64; // global_counts 删除次数
        let mut pq_max = 0usize; // 堆峰值长度
        let mut n_refill = 0u64; // refill_heap 调用次数（O(n) 全表扫描）
        let mut t_refill = 0f64; // refill 累计耗时（含 select 内与 GC 分支）
        let mut n_active = 0u64; // 命中活跃 shard 的次数
        let mut n_shard_apply = 0u64; // apply_merge 调用总次数（含 non-active 跳过）
        let t_merge_loop = std::time::Instant::now();
        const REPORT_EVERY: usize = 5000;

        // ---------- 自适应冻结间隔（2026-11-05 性能剖析驱动）----------
        // 实测：10k merges 的 merge 阶段有 59% 时间花在 `gc_freeze`，而它每次都要
        // 重建**全量倒排索引**（1.89 亿 word）。m=2 时前期几乎冻结不到 word
        // （B2 统计：189,063,238 unique words 里只有 309 个冻结）⟹ 这次重建是纯浪费。
        // 语义上冻结只删除"所有 pair 频次 < m"的 word（永远不可能再被选中），
        // 因此**推迟冻结不改变合并结果**，只影响内存回收时机。
        // 策略：按上一轮的冻结产率自适应调整间隔（产率低 ⟹ 拉长到 2 倍，上限
        // MAX_GC_INTERVAL；产率高 ⟹ 回到基线 500），保证死词不被无限堆积。
        let mut gc_interval = GC_INTERVAL_BASE;
        let mut next_gc_at = GC_INTERVAL_BASE;
        let mut gc_runs = 0usize;
        let mut gc_frozen_total = 0usize;

        for i in 0..num_merges {
            // R6-1：每轮 merge 是主检查点（优化后约 26ms/轮）。
            ck!(i);
            // M2：原实现用 `best_pair_u64 = 0` 作"无候选"哨兵，但 pack_pair(0,0) 也等于 0
            // （即两个 NUL 字节构成的**合法** pair）。语料含 NUL 时会被误判为"无候选"
            // 而提前终止，且 end_reason 写成 min_frequency_terminated（误导：真实原因是
            // 哨兵冲突而非频次不足）。改用 Option<u64> 后两者语义彻底分离。
            let mut best: Option<(u64, u64)> = None; // (pair, count)
            let t_blk = std::time::Instant::now();

            loop {
                if let Some(top) = pq.pop() {
                    if let Some(&actual_count) = global_counts.get(&top.pair) {
                        if actual_count == top.count {
                            best = Some((top.pair, actual_count));
                            break;
                        }
                    }
                } else {
                    // 堆空：从 global_counts 重新提取 top-K
                    let t_r = std::time::Instant::now();
                    refill_heap(&global_counts, &mut pq, HEAP_TOP_K, cancel, i)?;
                    t_refill += t_r.elapsed().as_secs_f64();
                    n_refill += 1;
                    if pq.is_empty() {
                        break;
                    }
                }
            }

            t_select += t_blk.elapsed().as_secs_f64();

            // 无候选 pair（堆已空且 global_counts 无更多条目）
            let Some((best_pair_u64, best_count)) = best else {
                actual_merges = i;
                end_reason = "no_more_pairs";
                info!(
                    "已无可合并的 pair，提前终止训练！当前真实词表合并进度: {}",
                    i
                );
                break;
            };

            // 【修改点 4】：基于阈值的截断防御
            if best_count < self.trainer_config.min_frequency as u64 {
                actual_merges = i;
                end_reason = "min_frequency_terminated";
                info!(
                    "最高词频 {} 低于阈值 {}，提前终止训练！当前真实词表合并进度: {}",
                    best_count, self.trainer_config.min_frequency, i
                );
                break;
            }

            let best_pair = unpack_pair(best_pair_u64); // (u32, u32)
            let new_id = 256 + i as u32;

            self.state
                .merges
                .insert((best_pair.0 as i32, best_pair.1 as i32), new_id as i32);
            let v0 = &self.state.vocab[&(best_pair.0 as i32)];
            let v1 = &self.state.vocab[&(best_pair.1 as i32)];
            let mut new_bytes = Vec::with_capacity(v0.len() + v1.len());
            new_bytes.extend_from_slice(v0);
            new_bytes.extend_from_slice(v1);
            self.state.vocab.insert(new_id as i32, new_bytes);

            // O(1) 鉴定活跃的 Shard：直接查 pair_head 倒排索引即可
            for j in 0..shards_count {
                shard_active[j] = shards[j].pair_head.contains_key(&best_pair_u64);
            }

            // 【完全体降临】：O(1) 提取受影响的词，并在微小数组内极速合并！
            let t_blk = std::time::Instant::now();
            shards
                .par_iter_mut()
                .zip(shard_active.par_iter())
                .zip(thread_deltas.par_iter_mut())
                .try_for_each(|((shard, &active), deltas)| {
                    ck_fine!(i);
                    deltas.clear();
                    if active {
                        shard.apply_merge(best_pair_u64, new_id, deltas)?;
                    }
                    Ok::<(), BbpeError>(())
                })?;
            t_apply += t_blk.elapsed().as_secs_f64();
            n_shard_apply += shards_count as u64;
            n_active += shard_active.iter().filter(|x| **x).count() as u64;

            // 极速注入 Deltas (使用 i64 防止大语料下溢出)
            let t_blk = std::time::Instant::now();
            for deltas in &thread_deltas {
                n_delta += deltas.len() as u64;
                for (&pair, &change) in deltas {
                    if change == 0 {
                        continue;
                    }
                    let entry = global_counts.entry(pair).or_insert(0);
                    if change > 0 {
                        *entry += change as u64;
                    } else {
                        *entry = entry.saturating_sub((-change) as u64);
                    }

                    let new_count = *entry;
                    if new_count == 0 {
                        global_counts.remove(&pair);
                        n_remove += 1;
                    } else {
                        pq.push(HeapEntry {
                            count: new_count,
                            pair,
                        });
                        n_push += 1;
                        if new_count < m {
                            n_push_below_m += 1;
                        }
                    }
                }
            }

            pq_max = pq_max.max(pq.len());

            // 【Step 2】：Lazy Heap GC —— 堆超过阈值时重建为 top-K (不再全量)
            if pq.len() > HEAP_GC_THRESHOLD {
                let t_r = std::time::Instant::now();
                refill_heap(&global_counts, &mut pq, HEAP_TOP_K, cancel, i)?;
                t_refill += t_r.elapsed().as_secs_f64();
                n_refill += 1;
                // shrink_to_fit 是 O(n) 全量拷贝，而重建只读 global_counts、
                // 并不改变条目数 ⟹ 只在容量明显虚高时才收回（此前每次都做，
                // 相当于给 96s 的重建再叠一次全表拷贝）。
                if global_counts.capacity() > global_counts.len() * 3 / 2 {
                    global_counts.shrink_to_fit();
                }
            }

            pb.inc(1);
            // ==========================================
            // 【终极防线】：每一轮合并结束后，让各个分片自行清理内存垃圾
            // ==========================================
            // ==========================================
            // 【替换】：加入定期物理降落机制
            // 每发生 500 次全局合并，强制触发一次全员深度 GC 与物理缩容！
            // ==========================================
            t_inject += t_blk.elapsed().as_secs_f64();
            let t_blk = std::time::Instant::now();
            if i + 1 >= next_gc_at {
                // B3: 深度 GC + 渐进冻结 (直接查 global_counts，避免构建大 HashSet)
                let words_before: usize = shards.iter().map(|s| s.word_starts.len()).sum();
                let total_frozen: usize = shards
                    .par_iter_mut()
                    .map(|shard| {
                        ck_fine!(i);
                        shard.gc_freeze(&global_counts, m)
                    })
                    .try_reduce(|| 0usize, |a, b| Ok(a + b))?;
                gc_runs += 1;
                gc_frozen_total += total_frozen;
                let ratio = total_frozen as f64 / words_before.max(1) as f64;
                let prev = gc_interval;
                gc_interval = next_gc_interval(gc_interval, ratio);
                next_gc_at = i + 1 + gc_interval;
                if total_frozen > 0 {
                    info!(
                        "B3 渐进冻结 @{}: 本轮冻结 {} words ({:.3}% of {}) ⟹ 下次间隔 {} (原 {})",
                        i,
                        total_frozen,
                        ratio * 100.0,
                        words_before,
                        gc_interval,
                        prev
                    );
                }
            } else {
                // 平时仅做容量水位线保护
                shards.par_iter_mut().try_for_each(|shard| {
                    ck_fine!(i);
                    shard.maybe_gc()
                })?;
            }
            t_gc += t_blk.elapsed().as_secs_f64();

            let done = i + 1;
            if done % REPORT_EVERY == 0 || done == num_merges {
                let el = t_merge_loop.elapsed().as_secs_f64();
                let per = el / done as f64;
                info!(
                    "merges {done}/{num_merges} | 已用 {el:.0}s | {:.1} ms/merge | ETA {:.0}s | select {:.1}% apply {:.1}% inject {:.1}% gc {:.1}% | avg 活跃shard {:.1}/{shards_count} | delta/merge {:.0}",
                    per * 1000.0,
                    per * (num_merges - done) as f64,
                    100.0 * t_select / el,
                    100.0 * t_apply / el,
                    100.0 * t_inject / el,
                    100.0 * t_gc / el,
                    n_active as f64 / done as f64,
                    n_delta as f64 / done as f64,
                );
            }
        }

        {
            let el = t_merge_loop.elapsed().as_secs_f64().max(1e-9);
            info!(
                "[profile] merge 阶段分解（占 merge 总耗时）：select {:.1}% | apply {:.1}% | \
                 inject {:.1}% | gc {:.1}% | 其它 {:.1}%",
                100.0 * t_select / el,
                100.0 * t_apply / el,
                100.0 * t_inject / el,
                100.0 * t_gc / el,
                100.0 * (el - t_select - t_apply - t_inject - t_gc) / el,
            );
            info!(
                "[profile] inject 细节：delta {:.1}M | push {:.1}M（其中 <m 的 {:.1}M = {:.1}%）| remove {:.1}M | 堆峰值 {:.1}M | refill {} 次 共 {:.1}s",
                n_delta as f64 / 1e6,
                n_push as f64 / 1e6,
                n_push_below_m as f64 / 1e6,
                100.0 * n_push_below_m as f64 / n_push.max(1) as f64,
                n_remove as f64 / 1e6,
                pq_max as f64 / 1e6,
                n_refill,
                t_refill
            );
            info!(
                "[profile] GC 次数 {} | 累计冻结 {} words | 末次间隔 {}",
                gc_runs, gc_frozen_total, gc_interval
            );
            info!(
                "[profile] 规模：{} merges | apply 调用 {:.1}M 次 | 命中活跃 shard 共 {:.1}M 次 | delta 共 {:.1}M 条",
                actual_merges,
                n_shard_apply as f64 / 1e6,
                n_active as f64 / 1e6,
                n_delta as f64 / 1e6
            );
        }
        pb_guard.finished = true;
        pb.finish_with_message("训练完成!");
        info!(
            "[bench] 阶段3(MapReduce合并) 耗时: {:.3}s",
            bench_t_phase.elapsed().as_secs_f64()
        );
        info!("[bench] 总耗时: {:.3}s", bench_t0.elapsed().as_secs_f64());

        // 模型契约 v1 §7.1：把训练参数/结果写入 meta（保存时随模型落盘）
        {
            let meta = self.state.meta_mut();
            meta.train = Some(TrainMeta {
                vocab_size: self.trainer_config.vocab_size,
                min_frequency: self.trainer_config.min_frequency,
                actual_merges,
                end_reason: end_reason.to_string(),
                max_chunk_bytes: self.runtime_config.max_chunk_bytes,
                // m6：训练产物本身不是截断得来的
                truncated_from: None,
            });
            if let Some(s) = &stats {
                meta.corpus = Some(s.snapshot_meta(reader.kind()));
            }
        }
        if let Some(s) = &stats {
            let (total_lines, total_bytes, skipped, replaced) = s.snapshot();
            info!(
                "corpus: total_lines={} total_bytes={} skipped_lines={} replacement_chars={}",
                total_lines, total_bytes, skipped, replaced
            );
        }

        Ok(())
    }

    // 3. 极度舒适的 Save 转发！(替换掉原本 20 多行的拼装代码)
    pub fn save(&self, filepath: &str) -> Result<(), BbpeError> {
        self.state.save(filepath)?;
        info!("模型已成功保存至: {}", filepath);
        Ok(())
    }

    /// 按保存选项写出模型（`no_meta` 等）。行为见 [`BpeState::save_with_options`]。
    ///
    /// M3：训练器此前只有 `save()`，无法产出"可逐字节比对"的 no_meta 模型，
    /// 只能绕过 trainer 直接操作 `state`；补上以保持 API 对称。
    pub fn save_with_options(&self, filepath: &str, opts: SaveOptions) -> Result<(), BbpeError> {
        self.state.save_with_options(filepath, opts)?;
        info!("模型已成功保存至: {}", filepath);
        Ok(())
    }
}

#[cfg(test)]
mod gc_policy_tests {
    //! 自适应冻结间隔策略的定点测试（2026-11-06 补）。
    //!
    //! 背景：该策略把「每 500 merges 全量 gc_freeze」改成按冻结产率伸缩间隔，
    //! 实测把 9.96GB/150k merges 的训练从 10974s 压到 4503s。正确性由端到端
    //! 等价性保证（两次训练 vocab/merges 的 sha256 相同），但"间隔怎么变"这条
    //! 策略本身需要定点回归，否则以后调阈值没人能发现退化。
    use super::{GC_INTERVAL_BASE, GC_INTERVAL_MAX, next_gc_interval};

    #[test]
    fn low_yield_doubles_until_cap() {
        let mut iv = GC_INTERVAL_BASE;
        let mut seq = vec![iv];
        for _ in 0..10 {
            iv = next_gc_interval(iv, 0.0); // 一个都没冻结
            seq.push(iv);
        }
        assert_eq!(&seq[..4], &[500, 1000, 2000, 4000]);
        assert_eq!(
            *seq.last().unwrap(),
            GC_INTERVAL_MAX,
            "必须封顶，不能无限翻倍"
        );
        assert!(seq.windows(2).all(|w| w[0] <= w[1]), "低产率下间隔单调不减");
    }

    #[test]
    fn yield_at_low_threshold_keeps_interval() {
        // 边界取严格小于：0.003 不算"低产率"，应维持当前节奏
        assert_eq!(next_gc_interval(2000, 0.003), 2000);
    }

    #[test]
    fn high_yield_falls_back_to_baseline() {
        assert_eq!(next_gc_interval(GC_INTERVAL_MAX, 0.5), GC_INTERVAL_BASE);
        // 阈值也是严格大于：0.01 不触发回落
        assert_eq!(next_gc_interval(2000, 0.01), 2000);
        assert_eq!(next_gc_interval(2000, 0.0101), GC_INTERVAL_BASE);
    }

    #[test]
    fn mid_band_keeps_interval() {
        for iv in [GC_INTERVAL_BASE, 2000, 8000, GC_INTERVAL_MAX] {
            assert_eq!(next_gc_interval(iv, 0.005), iv);
        }
    }

    #[test]
    fn no_overflow_at_cap() {
        assert_eq!(next_gc_interval(usize::MAX, 0.0), GC_INTERVAL_MAX);
        assert_eq!(next_gc_interval(GC_INTERVAL_MAX, 0.0), GC_INTERVAL_MAX);
    }

    #[test]
    fn zero_interval_does_not_hang() {
        // 防御性：中间带下 prev=0 会被抬到 1，避免 next_gc_at 原地踏步
        assert_eq!(next_gc_interval(0, 0.005), 1);
    }

    #[test]
    fn non_finite_yield_keeps_interval() {
        // NaN 与任何比较都为 false ⟹ 落到"保持"分支，不会误触发翻倍/回落
        assert_eq!(next_gc_interval(4000, f64::NAN), 4000);
    }
}

#[cfg(test)]
mod corpus_guard_tests {
    //! G4：语料**整体不可用**时必须显式失败。
    //!
    //! 回归背景（2026-11-08）：把纯文本当 `--format jsonl`（CLI 默认值）读入，
    //! 每一行都解析失败被跳过，训练却一路返回 Ok，产出一个"256 个基础字节 +
    //! 0 条 merges"的空模型，`inspect --validate-only` 也判它合法。用户拿到的是
    //! 一个看起来正常、实际什么都编码不了的模型文件。
    use super::*;
    use crate::error::BbpeError;
    use crate::reader::{CorpusReader, CorpusStats};
    use std::sync::Arc;

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("bbpe_guard_{}_{name}.txt", std::process::id()));
        std::fs::write(&p, content).expect("写临时语料失败");
        p
    }

    /// 走"带统计"的训练路径（CLI / Python 的真实路径）。
    /// 无计数器的 `train()` 拿不到 total/skipped，只能给通用文案。
    fn train_expect_empty_corpus(reader: &CorpusReader, path: &std::path::Path) -> String {
        let mut trainer = BBPETrainer::new(TrainerConfig::new(300), None).expect("构造 trainer");
        let err = trainer
            .train_with_stats(reader, Arc::new(CorpusStats::default()))
            .expect_err("0 行可用必须报错");
        // 失败后不得留下"可保存"的训练态
        assert!(!trainer.is_trained(), "失败后不得标记为已训练");
        let _ = std::fs::remove_file(path);
        match err {
            BbpeError::EmptyCorpus(msg) => msg,
            other => panic!("错误类型应为 EmptyCorpus，实际: {other:?}"),
        }
    }

    #[test]
    fn plain_text_as_jsonl_fails_not_empty_model() {
        let p = write_temp("jsonl", "hello world\n你好 世界\n");
        let reader = CorpusReader::JsonlStaticText {
            filepath: p.to_string_lossy().into_owned(),
        };
        let msg = train_expect_empty_corpus(&reader, &p);
        assert!(msg.contains("解析失败"), "应指出是逐行解析失败: {msg}");
        assert!(
            msg.contains("--format text"),
            "错误里必须给出 CLI 的可操作处置: {msg}"
        );
        assert!(
            msg.contains("TextFileReader"),
            "错误里必须给出 Python 的可操作处置: {msg}"
        );
    }

    #[test]
    fn empty_file_reports_empty_input() {
        let p = write_temp("empty", "");
        let reader = CorpusReader::Text {
            filepath: p.to_string_lossy().into_owned(),
        };
        let msg = train_expect_empty_corpus(&reader, &p);
        assert!(msg.contains("输入文件为空"), "应指出文件为空: {msg}");
        assert!(
            !msg.contains("--format text"),
            "空文件与格式无关，不得把用户引向改 --format: {msg}"
        );
    }

    #[test]
    fn blank_lines_reports_blank_content() {
        let p = write_temp("blank", "\n\n\n");
        let reader = CorpusReader::Text {
            filepath: p.to_string_lossy().into_owned(),
        };
        let msg = train_expect_empty_corpus(&reader, &p);
        assert!(msg.contains("空行"), "应指出全部是空行: {msg}");
        assert!(
            !msg.contains("--format text"),
            "空行不是格式问题，不得把用户引向改 --format: {msg}"
        );
    }

    #[test]
    fn same_content_trains_with_text_reader() {
        let p = write_temp("text", "hahaha\nhahaha\nhahaha\n");
        let reader = CorpusReader::Text {
            filepath: p.to_string_lossy().into_owned(),
        };
        let mut trainer = BBPETrainer::new(TrainerConfig::new(300), None).expect("构造 trainer");
        trainer.train(&reader).expect("text reader 应正常训练");
        assert!(trainer.is_trained(), "正常语料训练后应处于已训练态");
        let _ = std::fs::remove_file(p);
    }
}

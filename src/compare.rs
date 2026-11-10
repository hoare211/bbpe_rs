//! 合并序列对照（`bbpe compare` 的**库实现**）。
//!
//! 回答"采样率 / 语料规模是否足够"这个核心问题：大语料用采样代替全量训练，
//! 正确性依赖"头部 pair 频次在采样下等比例缩放 ⟹ 相对次序不变"。这个论证
//! 必须有**端到端**（而非统计量层面）的验证 —— 直接比两条训练出来的 merges
//! 序列，看它们在前 N 项上是否一致、在哪里分叉。
//!
//! ⚠️ 关键方法学：**跨模型比较 merges 必须在「字节空间」进行，id 空间不可比。**
//!    本项目的 `new_id = 256 + i` 由合并次序定义 ⟹ 一旦两条序列在第 i 位分叉，
//!    其后所有 token id 的含义就整体错位，id 空间的重合率会崩到 ~6%，
//!    但那是**编号错位**造成的假象，不是词表内容的差异。
//!    正确做法：用 `vocab[id]` 把每个 merge 解码成字节串对 `(bytes_a, bytes_b)` 再比。
//!
//! 三个指标：
//!   1. **逐位一致率（positional）**：第 i 次合并的字节串对是否相同（最严格口径）。
//!      第一个分叉位置 = 两条路径真正开始偏离的点。
//!   2. **集合重合率（set，次序无关）**：A 的前 N 项有多少也出现在 B 的前 N 项。
//!   3. **id 空间重合率（对照）**：仅用于暴露上述编号错位效应，**不作为结论依据**。
//!
//! 判据：分叉点越靠后越好；若分叉点在实际使用的 K 之后，则两模型对下游等价。
//!
//! 本模块是唯一实现：`cli::compare`（打印）与 Python 绑定（`compare_models` /
//! `BBPETokenizer.compare`）都调它，避免各写一份指标算法后口径漂移。

use crate::tokenizer::BpeState;
use serde::Serialize;
use std::collections::HashSet;

/// 一次合并的「字节空间」表示：两边各自的完整字节串。
pub type DecodedMerge = (Vec<u8>, Vec<u8>);

/// `compare` 默认的分段观测点（与 CLI `--at` 的默认值一致）。
pub const DEFAULT_AT: [usize; 5] = [1_000, 10_000, 50_000, 100_000, 150_000];

/// 把 `merges: (a,b) -> new_id` 还原成"按合并轮次排序"的序列。
/// `new_id = 256 + i` ⟹ 按 `new_id` 升序即为第 0,1,2,... 次合并。
pub fn merge_sequence(state: &BpeState) -> Vec<(i32, i32)> {
    let mut seq: Vec<(i32, (i32, i32))> = state
        .merges
        .iter()
        .map(|(&(a, b), &nid)| (nid, (a, b)))
        .collect();
    seq.sort_unstable();
    seq.into_iter().map(|(_, p)| p).collect()
}

/// 解码：id 对 → 字节串对。缺 id 时回退为 id 文本（不应发生）。
pub fn decode_merges(state: &BpeState, seq: &[(i32, i32)]) -> Vec<DecodedMerge> {
    seq.iter()
        .map(|&(a, b)| {
            (
                state
                    .vocab
                    .get(&a)
                    .cloned()
                    .unwrap_or_else(|| a.to_string().into_bytes()),
                state
                    .vocab
                    .get(&b)
                    .cloned()
                    .unwrap_or_else(|| b.to_string().into_bytes()),
            )
        })
        .collect()
}

/// 便于展示：尽量按 UTF-8 显示，否则退化为字节转义。
pub fn show_bytes(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => format!("{s:?}"),
        Err(_) => format!("{bytes:02x?}"),
    }
}

/// 首个分叉点的细节（渲染后的字符串，供 CLI 直接打印、Python 直接读）。
#[derive(Debug, Clone, Serialize)]
pub struct DivergenceDetail {
    /// 0-based 的合并轮次。
    pub index: usize,
    /// A 侧该次合并的两个字节串（已渲染）。
    pub a: (String, String),
    /// B 侧该次合并的两个字节串（已渲染）。
    pub b: (String, String),
}

/// 全体逐位一致率。
#[derive(Debug, Clone, Serialize)]
pub struct PositionalMatch {
    /// 逐位相同的项数。
    pub matched: usize,
    /// 参与比较的项数（两侧 merges 数的较小者）。
    pub compared: usize,
    /// `matched / compared`。
    ///
    /// `compared == 0`（至少一侧 merges 为 0 ⟹ 无共同前缀可比较）时为 `None`：
    /// 既不是 0%（那是"完全不一致"）也不是 100%，而是**无从比较**。
    /// 用 `NaN` 表达会同时坏掉两件事 —— JSON 序列化（`serde_json` 拒绝 NaN）
    /// 与 CLI 打印（`NaN%`）。
    pub ratio: Option<f64>,
}

/// 某个分段观测点上的三项指标。
#[derive(Debug, Clone, Serialize)]
pub struct SegmentMetric {
    /// 实际参与比较的前缀长度（已按两侧较小 merges 数钳制）。
    pub k: usize,
    /// 逐位一致率（最严格口径）。
    pub positional_ratio: f64,
    /// 集合重合率（次序无关，字节空间）。
    pub set_overlap_ratio: f64,
    /// id 空间重合率（**仅作对照**，不可作结论依据）。
    pub id_space_ratio: f64,
}

/// 一次对照的完整结果（纯数据；判定策略留给调用方）。
#[derive(Debug, Clone, Serialize)]
pub struct CompareReport {
    /// A 侧的 merges 条数。
    pub merges_a: usize,
    /// B 侧的 merges 条数。
    pub merges_b: usize,
    /// 参与比较的项数 = `min(merges_a, merges_b)`。
    pub compared: usize,
    /// 首个分叉位置（0-based）；`None` = 前 `compared` 项逐位完全一致。
    pub first_divergence: Option<usize>,
    /// 分叉点细节；与 `first_divergence` 同生共死。
    pub divergence: Option<DivergenceDetail>,
    /// 全体逐位一致率。
    pub positional: PositionalMatch,
    /// 各分段观测点的指标（仅包含 `k > 0` 的项）。
    pub at: Vec<SegmentMetric>,
}

/// 对照两个 state 的合并序列（唯一的指标实现）。
///
/// `at` 为分段观测点（如 `[1000, 10_000]`），超过两侧较小 merges 数的会被钳制，
/// 钳制后为 0 的点直接跳过。
pub fn compare_states(a: &BpeState, b: &BpeState, at: &[usize]) -> CompareReport {
    let ra = merge_sequence(a); // id 空间（仅作对照）
    let rb = merge_sequence(b);
    let ma = decode_merges(a, &ra); // 字节空间（结论依据）
    let mb = decode_merges(b, &rb);

    let n = ma.len().min(mb.len());

    // ① 逐位一致：第一个分叉位置（字节空间）
    let mut first_diff: Option<usize> = None;
    let mut pos_match = 0usize;
    for i in 0..n {
        if ma[i] == mb[i] {
            pos_match += 1;
        } else if first_diff.is_none() {
            first_diff = Some(i);
        }
    }
    let divergence = first_diff.map(|i| DivergenceDetail {
        index: i,
        a: (show_bytes(&ma[i].0), show_bytes(&ma[i].1)),
        b: (show_bytes(&mb[i].0), show_bytes(&mb[i].1)),
    });

    // ② 分段指标
    let mut segments = Vec::new();
    for &k in at {
        let k = k.min(n);
        if k == 0 {
            continue;
        }
        let pm = (0..k).filter(|&i| ma[i] == mb[i]).count();
        let set_a: HashSet<&DecodedMerge> = ma[..k].iter().collect();
        let set_b: HashSet<&DecodedMerge> = mb[..k].iter().collect();
        let inter = set_a.intersection(&set_b).count();
        let id_a: HashSet<(i32, i32)> = ra[..k].iter().copied().collect();
        let id_b: HashSet<(i32, i32)> = rb[..k].iter().copied().collect();
        let id_inter = id_a.intersection(&id_b).count();
        segments.push(SegmentMetric {
            k,
            positional_ratio: pm as f64 / k as f64 * 100.0,
            set_overlap_ratio: inter as f64 / k as f64 * 100.0,
            id_space_ratio: id_inter as f64 / k as f64 * 100.0,
        });
    }

    CompareReport {
        merges_a: ma.len(),
        merges_b: mb.len(),
        compared: n,
        first_divergence: first_diff,
        divergence,
        positional: PositionalMatch {
            matched: pos_match,
            compared: n,
            ratio: (n > 0).then(|| pos_match as f64 / n as f64),
        },
        at: segments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// 构造一个最小可用的 state：vocab = 256 基础字节 + 若干 merge 产物。
    ///
    /// 只用 `compare_states` 需要的字段；`merges_fast` 由 `truncate_to_k`/
    /// `rebuild_merges_fast` 维护，这里不参与比较逻辑。
    fn state_from(merges: &[((i32, i32), i32)]) -> BpeState {
        let mut vocab: HashMap<i32, Vec<u8>> = (0..256).map(|i| (i, vec![i as u8])).collect();
        let mut map = HashMap::new();
        for &(pair, nid) in merges {
            let (a, b) = pair;
            let mut bytes = vocab.get(&a).cloned().unwrap_or_default();
            bytes.extend(vocab.get(&b).cloned().unwrap_or_default());
            vocab.insert(nid, bytes);
            map.insert(pair, nid);
        }
        BpeState {
            vocab,
            merges: map,
            merges_fast: rustc_hash::FxHashMap::default(),
            special_tokens: HashMap::new(),
            pre_tokenizer_pattern: crate::pre_tokenizer::DEFAULT_PATTERN.to_string(),
            meta: None,
            format: None,
            format_version: None,
            legacy_pattern_fallback: false,
        }
    }

    fn sample_state() -> BpeState {
        // 256+0 = 'h'+'i'? 用基础字节组合出可读的合并链：("a","b") -> 256, ...
        state_from(&[((97, 98), 256), ((256, 99), 257), ((100, 101), 258)])
    }

    #[test]
    fn identical_when_compared_with_itself() {
        let s = sample_state();
        let r = compare_states(&s, &s, &DEFAULT_AT);
        assert_eq!(r.compared, 3);
        assert_eq!(r.first_divergence, None);
        assert!(r.divergence.is_none());
        assert_eq!(r.positional.matched, 3);
        assert_eq!(r.positional.ratio, Some(1.0));
        // 全部观测点（钳制到 3）都应 100%
        assert!(!r.at.is_empty());
        for seg in &r.at {
            assert_eq!(seg.k, 3);
            assert!((seg.positional_ratio - 100.0).abs() < 1e-9);
            assert!((seg.set_overlap_ratio - 100.0).abs() < 1e-9);
            assert!((seg.id_space_ratio - 100.0).abs() < 1e-9);
        }
    }

    #[test]
    fn divergence_index_and_detail_are_exact() {
        let a = sample_state();
        // 只改最后一次合并的目标 pair ⟹ 分叉点必然是第 2 次（0-based）
        let b = state_from(&[((97, 98), 256), ((256, 99), 257), ((100, 102), 258)]);
        let r = compare_states(&a, &b, &DEFAULT_AT);
        assert_eq!(r.first_divergence, Some(2));
        let d = r.divergence.expect("必须带分叉细节");
        assert_eq!(d.index, 2);
        // A 的第 3 次合并是 ("d","e")，B 的是 ("d","f")
        assert_eq!(d.a, ("\"d\"".to_string(), "\"e\"".to_string()));
        assert_eq!(d.b, ("\"d\"".to_string(), "\"f\"".to_string()));
        assert_eq!(r.positional.matched, 2);
    }

    #[test]
    fn ratio_is_none_when_nothing_to_compare() {
        let a = sample_state();
        let empty = state_from(&[]);
        let r = compare_states(&a, &empty, &DEFAULT_AT);
        assert_eq!(r.compared, 0);
        assert_eq!(r.first_divergence, None);
        assert_eq!(
            r.positional.ratio, None,
            "0 项可比必须是 None：NaN 既不能被 serde_json 序列化，也会打印成 NaN%"
        );
        assert!(r.at.is_empty(), "k 被钳制到 0 的观测点必须跳过");
        // 空 vs 空同样成立，且报告必须可序列化（NaN 会让 to_value 直接失败）
        let r2 = compare_states(&empty, &empty, &DEFAULT_AT);
        assert_eq!(r2.positional.ratio, None);
        assert!(
            serde_json::to_value(&r2).is_ok(),
            "报告必须可 JSON 序列化（Python 绑定走 serde）"
        );
    }

    #[test]
    fn truncated_model_matches_prefix() {
        // cut 的语义承诺：前 k 次合并与原模型**逐位相同**
        let a = sample_state();
        let mut b = sample_state();
        b.truncate_to_k(2);
        let r = compare_states(&a, &b, &DEFAULT_AT);
        assert_eq!(r.compared, 2);
        assert_eq!(r.first_divergence, None);
        assert_eq!(r.positional.matched, 2);
    }
}

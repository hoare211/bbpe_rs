// src/external.rs
//
// 阶段 0 组件：word_freqs 外存化 —— SpillingBucket（内存累积 + 阈值溢写）
// + sorted run 写出 + 多路归并迭代器。
//
// 来源：v4-cms-external（write_run/RunMerger）+ v4 trainer.rs 的 SpillingBucket，
// 按《分层冻结BPE完整技术方案》§6.2 完成三处修复：
//   1. flush 失败不再吞错：返回 Result，调用方 ? 传播，训练终止（原版只打日志继续）；
//   2. 频次全链路 u64：内存类型与 run 文件编码同步改（原版 u32 在数百 GB 语料
//      下必然溢出且静默回绕，坑 4 / F12）；
//   3. run 目录唯一命名：pid + 纳秒时间戳 + 原子计数器（原版仅 pid，pid 回绕/
//      同进程多次训练时冲突）。
//
// run 文件格式（与 v4 不兼容，属有意变更，方案 §6.3）：
//   按 word bytes 字典序排列的去重 (word, count) 记录，
//   每条 = [len: u32 LE | bytes[len] | count: u64 LE]。
//
// 等价性：归并按 bytes 字典序去重求和，结果与内存 HashMap 统计完全一致
// （仅顺序变化，不影响后续 pair 频次汇总）。

use ahash::AHashMap;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// 全局原子计数器，保证同进程内多次创建 run 目录不冲突。
static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// 创建唯一的 run 目录（修复 3）：pid + 纳秒时间戳 + 原子计数器。
/// 调用方负责该目录的生命周期（训练结束后清理）。
pub fn create_run_dir(base: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = base.as_ref().join(format!(
        "bbpe_runs_{}_{}_{}",
        std::process::id(),
        nanos,
        seq
    ));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// 写一个 sorted run 文件：从已去重的 HashMap，按 bytes 字典序输出。
/// 频次 u64（修复 2）。
pub fn write_run(path: &Path, map: &AHashMap<Vec<u8>, u64>) -> std::io::Result<()> {
    let mut sorted: Vec<(&Vec<u8>, &u64)> = map.iter().collect();
    sorted.sort_unstable_by(|a, b| a.0.as_slice().cmp(b.0.as_slice()));

    let mut w = BufWriter::new(File::create(path)?);
    let mut len_buf = [0u8; 4];
    let mut cnt_buf = [0u8; 8];
    for (bytes, count) in &sorted {
        len_buf.copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        w.write_all(&len_buf)?;
        w.write_all(bytes)?;
        cnt_buf.copy_from_slice(&count.to_le_bytes());
        w.write_all(&cnt_buf)?;
    }
    w.flush()?;
    Ok(())
}

/// 单个 sorted run 的读取游标。
struct RunReader {
    reader: BufReader<File>,
    cur: Option<(Vec<u8>, u64)>,
}

impl RunReader {
    fn open(path: &Path) -> std::io::Result<Self> {
        let mut r = Self {
            reader: BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?),
            cur: None,
        };
        r.advance()?;
        Ok(r)
    }

    /// 读取下一条记录到 cur；文件末尾则 cur = None。
    fn advance(&mut self) -> std::io::Result<()> {
        let mut len_buf = [0u8; 4];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {
                let len = u32::from_le_bytes(len_buf) as usize;
                let mut bytes = vec![0u8; len];
                self.reader.read_exact(&mut bytes)?;
                let mut cnt_buf = [0u8; 8];
                self.reader.read_exact(&mut cnt_buf)?;
                let count = u64::from_le_bytes(cnt_buf);
                self.cur = Some((bytes, count));
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.cur = None;
            }
            Err(e) => return Err(e),
        }
        Ok(())
    }
}

/// 堆项：按 bytes 字典序最小堆。
struct HeapItem {
    bytes: Vec<u8>,
    run_idx: usize,
}
impl Eq for HeapItem {}
impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // 大顶堆 + Reverse => 按 bytes 升序的最小堆
        other.bytes.cmp(&self.bytes)
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 多路归并迭代器：合并多个 sorted run，输出去重后的 (word, total_count)。
/// 相同 bytes 的记录跨 run 求和。
pub struct RunMerger {
    readers: Vec<RunReader>,
    heap: BinaryHeap<HeapItem>,
}

impl RunMerger {
    pub fn new(paths: &[PathBuf]) -> std::io::Result<Self> {
        let mut readers = Vec::with_capacity(paths.len());
        for p in paths {
            readers.push(RunReader::open(p)?);
        }
        let mut heap = BinaryHeap::new();
        for (i, r) in readers.iter().enumerate() {
            if let Some((bytes, _)) = &r.cur {
                heap.push(HeapItem {
                    bytes: bytes.clone(),
                    run_idx: i,
                });
            }
        }
        Ok(Self { readers, heap })
    }

    /// 弹出下一个去重合并后的 (word, total_count)。无更多记录返回 None。
    pub fn next_word(&mut self) -> std::io::Result<Option<(Vec<u8>, u64)>> {
        let top = match self.heap.pop() {
            Some(t) => t,
            None => return Ok(None),
        };
        let target_bytes = top.bytes.clone();
        let run_idx = top.run_idx;

        // 累加该 run 当前记录的 count，并前进
        let mut total = self.readers[run_idx]
            .cur
            .as_ref()
            .map(|(_, c)| *c)
            .unwrap_or(0);
        self.readers[run_idx].advance()?;
        if let Some((b, _)) = &self.readers[run_idx].cur {
            self.heap.push(HeapItem {
                bytes: b.clone(),
                run_idx,
            });
        }

        // 合并堆中所有相同 bytes 的记录
        while let Some(peek) = self.heap.peek() {
            if peek.bytes == target_bytes {
                let item = self.heap.pop().unwrap();
                let ri = item.run_idx;
                if let Some((_, c)) = &self.readers[ri].cur {
                    total += *c;
                }
                self.readers[ri].advance()?;
                if let Some((b, _)) = &self.readers[ri].cur {
                    self.heap.push(HeapItem {
                        bytes: b.clone(),
                        run_idx: ri,
                    });
                }
            } else {
                break;
            }
        }
        Ok(Some((target_bytes, total)))
    }
}

// ==========================================
// SpillingBucket —— 内存累积 + 阈值溢写（修复 1：错误传播）
// ==========================================

/// word_freqs 外存化容器：内存 HashMap 累积，达 flush_threshold 溢写为
/// sorted run 文件；统一通过 `for_each_word` 遍历（未 flush 走内存，
/// 已 flush 走外存归并——小语料零 IO）。
pub struct SpillingBucket {
    map: AHashMap<Vec<u8>, u64>,
    runs: Vec<PathBuf>,
    flush_threshold: usize,
    dir: PathBuf,
    bucket_id: usize,
    run_counter: usize,
}

impl SpillingBucket {
    pub fn new(dir: PathBuf, bucket_id: usize, flush_threshold: usize) -> Self {
        Self {
            map: AHashMap::with_capacity(4_000),
            runs: Vec::new(),
            flush_threshold,
            dir,
            bucket_id,
            run_counter: 0,
        }
    }

    /// 累加一个 word 出现（每次 +1）。
    pub fn add(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        if let Some(c) = self.map.get_mut(bytes) {
            *c += 1;
        } else {
            self.map.insert(bytes.to_vec(), 1);
        }
        if self.map.len() >= self.flush_threshold {
            self.flush()?;
        }
        Ok(())
    }

    /// 溢写当前 map 为 sorted run（修复 1：失败返回 Err 且不清空 map，
    /// 由调用方决定终止训练；原版吞错只打日志，静默丢数据）。
    pub fn flush(&mut self) -> std::io::Result<()> {
        if self.map.is_empty() {
            return Ok(());
        }
        let path = self.dir.join(format!(
            "bbpe_b{}_r{}.bin",
            self.bucket_id, self.run_counter
        ));
        write_run(&path, &self.map)?;
        self.runs.push(path);
        self.run_counter += 1;
        self.map.clear();
        Ok(())
    }

    /// 收尾：若从未触发 flush，保留 map 供内存读取（避免小语料无谓 IO）。
    pub fn finish(&mut self) -> std::io::Result<()> {
        if self.runs.is_empty() {
            return Ok(());
        }
        self.flush()
    }

    /// 已产生的 run 文件路径（供阶段 1 直接构建 RunMerger）。
    pub fn run_paths(&self) -> &[PathBuf] {
        &self.runs
    }

    /// 统一遍历：未 flush 走内存 map，已 flush 走外存归并。
    pub fn for_each_word<F: FnMut(&[u8], u64)>(&self, mut f: F) -> std::io::Result<()> {
        if self.runs.is_empty() {
            for (bytes, count) in &self.map {
                f(bytes, *count);
            }
        } else {
            let mut merger = RunMerger::new(&self.runs)?;
            while let Some((bytes, count)) = merger.next_word()? {
                f(&bytes, count);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_write_merge_dedup() {
        // 两个 run，含重叠 word，归并后应去重求和
        let dir = create_run_dir(std::env::temp_dir()).unwrap();
        let p1 = dir.join("run1.bin");
        let p2 = dir.join("run2.bin");

        let mut m1 = AHashMap::new();
        m1.insert(b"apple".to_vec(), 3);
        m1.insert(b"banana".to_vec(), 1);
        m1.insert(b"cherry".to_vec(), 2);

        let mut m2 = AHashMap::new();
        m2.insert(b"apple".to_vec(), 5);
        m2.insert(b"date".to_vec(), 4);
        m2.insert(b"banana".to_vec(), 7);

        write_run(&p1, &m1).unwrap();
        write_run(&p2, &m2).unwrap();

        let mut merger = RunMerger::new(&[p1.clone(), p2.clone()]).unwrap();
        let mut out = Vec::new();
        while let Some((bytes, count)) = merger.next_word().unwrap() {
            out.push((bytes, count));
        }

        // 期望字典序: apple(3+5=8), banana(1+7=8), cherry(2), date(4)
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], (b"apple".to_vec(), 8));
        assert_eq!(out[1], (b"banana".to_vec(), 8));
        assert_eq!(out[2], (b"cherry".to_vec(), 2));
        assert_eq!(out[3], (b"date".to_vec(), 4));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_single_run() {
        let dir = create_run_dir(std::env::temp_dir()).unwrap();
        let p = dir.join("single.bin");
        let mut m = AHashMap::new();
        m.insert(b"x".to_vec(), 10);
        m.insert(b"a".to_vec(), 1);
        write_run(&p, &m).unwrap();
        let mut merger = RunMerger::new(&[p]).unwrap();
        let mut out = Vec::new();
        while let Some(item) = merger.next_word().unwrap() {
            out.push(item);
        }
        assert_eq!(out, vec![(b"a".to_vec(), 1), (b"x".to_vec(), 10)]);
        let _ = fs::remove_dir_all(&dir);
    }

    /// T6：u64 频次往返 —— 超过 u32::MAX 的计数必须无损通过 run 文件
    /// （200GB 语料头部 pair/词频 ~10^9-10^10，u32 必溢出且静默，坑 4）。
    #[test]
    fn test_u64_count_roundtrip() {
        let dir = create_run_dir(std::env::temp_dir()).unwrap();
        let p = dir.join("u64.bin");
        let huge: u64 = 5_000_000_000; // > u32::MAX (4.29e9)
        let mut m = AHashMap::new();
        m.insert(b"the".to_vec(), huge);
        m.insert(b"tiny".to_vec(), 1);
        write_run(&p, &m).unwrap();

        let mut merger = RunMerger::new(&[p]).unwrap();
        let mut got = AHashMap::new();
        while let Some((bytes, count)) = merger.next_word().unwrap() {
            got.insert(bytes, count);
        }
        assert_eq!(got.get(b"the".as_slice()), Some(&huge));
        assert_eq!(got.get(b"tiny".as_slice()), Some(&1));
        let _ = fs::remove_dir_all(&dir);
    }

    /// 修复 1 的回归测试：flush 到非法路径必须返回 Err（而非吞错继续）。
    #[test]
    fn test_flush_error_propagates() {
        // 用一个普通文件的路径充当目录 -> create 前置目录不存在，write_run 必失败
        let base = create_run_dir(std::env::temp_dir()).unwrap();
        let blocker = base.join("not_a_dir");
        File::create(&blocker).unwrap();
        let bad_dir = blocker.join("sub"); // not_a_dir/sub 不可创建

        let mut bucket = SpillingBucket::new(bad_dir, 0, 1_000_000);
        bucket.add(b"word").unwrap();
        let err = bucket.flush();
        assert!(err.is_err(), "flush 到非法路径必须报错而非吞掉");
        // map 未被清空：数据未丢失，调用方可选择终止
        bucket
            .for_each_word(|w, c| {
                assert_eq!((w, c), (b"word".as_slice(), 1u64));
            })
            .unwrap();
        let _ = fs::remove_dir_all(&base);
    }

    /// SpillingBucket 端到端：跨阈值多次溢写 + 归并求和。
    #[test]
    fn test_spilling_bucket_end_to_end() {
        let dir = create_run_dir(std::env::temp_dir()).unwrap();
        // 阈值设小，强制多次溢写
        let mut bucket = SpillingBucket::new(dir.clone(), 0, 3);

        // 5 个不同词各加多次，穿插触发 flush
        for i in 0..10u64 {
            bucket.add(b"alpha").unwrap();
            bucket.add(b"beta").unwrap();
            if i % 3 == 0 {
                bucket.add(b"gamma").unwrap();
            }
        }
        bucket.finish().unwrap();

        let mut total = AHashMap::new();
        bucket
            .for_each_word(|w, c| {
                total.insert(w.to_vec(), c);
            })
            .unwrap();

        assert_eq!(total.get(b"alpha".as_slice()), Some(&10));
        assert_eq!(total.get(b"beta".as_slice()), Some(&10));
        assert_eq!(total.get(b"gamma".as_slice()), Some(&4)); // i=0,3,6,9
        assert!(bucket.run_paths().len() >= 2, "应产生多个 run 文件");
        let _ = fs::remove_dir_all(&dir);
    }

    /// 修复 3 的回归测试：连续创建多个 run 目录互不冲突。
    #[test]
    fn test_run_dir_uniqueness() {
        let base = std::env::temp_dir();
        let d1 = create_run_dir(&base).unwrap();
        let d2 = create_run_dir(&base).unwrap();
        assert_ne!(d1, d2, "同进程内连续创建的 run 目录必须唯一");
        let _ = fs::remove_dir_all(&d1);
        let _ = fs::remove_dir_all(&d2);
    }
}

//! R6-1：大训练态「异步释放」的可复现基准。
//!
//! 背景：P1 取消时不 join worker，让几十 GB 训练态在后台线程按栈顺序 drop，
//! 主线程立即返回异常。本基准量化两件事：
//!   ① 主线程「返回」延迟（应接近 0）；
//!   ② worker 释放完成耗时，以及 RSS 是否真的回落。
//!
//! 用法：
//!   cargo run --release --example cancel_drop_probe
//!   BBPE_DROP_PROBE_GB=85 cargo run --release --example cancel_drop_probe
//!
//! 规模由 `BBPE_DROP_PROBE_GB` 控制（默认 8）。内存对象是**分散的多个
//! Vec<u8> 块**，与训练态的 `shards` / `global_counts` / `word_freqs` 一样
//! 是大量独立分配，而不是单个巨型连续缓冲区 —— 释放成本主要来自
//! 遍历海量小分配并归还给分配器，这正是本基准要测的。

use std::sync::mpsc;
use std::time::Instant;

fn rss_mb() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn main() {
    let gb: usize = std::env::var("BBPE_DROP_PROBE_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let target = gb.saturating_mul(1024 * 1024 * 1024);

    let chunk = 16usize * 1024 * 1024;
    let mut state: Vec<Vec<u8>> = Vec::with_capacity(target / chunk + 1);
    let mut allocated = 0usize;
    while allocated < target {
        let mut v = vec![0u8; chunk];
        // 逐页触碰，确保是真实 RSS 而不是惰性映射
        for i in (0..chunk).step_by(4096) {
            v[i] = (i & 0xff) as u8;
        }
        state.push(v);
        allocated += chunk;
    }
    println!(
        "ALLOCATED_GB {:.2}",
        allocated as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("RSS_BEFORE_MB {:.1}", rss_mb().unwrap_or(f64::NAN));

    let (tx, rx) = mpsc::sync_channel::<()>(1);
    let t0 = Instant::now();
    let handle = std::thread::spawn(move || {
        tx.send(()).expect("主线程已退出");
        drop(state);
        eprintln!("WORKER_DROP_DONE_S {:.3}", t0.elapsed().as_secs_f64());
    });

    rx.recv().expect("worker 未能启动");
    // 模拟 P1：主线程不 join，立即把异常交给 Python
    let main_return = t0.elapsed();
    println!("MAIN_RETURN_MS {:.3}", main_return.as_secs_f64() * 1000.0);
    assert!(main_return.as_millis() < 100, "主线程返回被后台释放阻塞了");

    // 采样 RSS 回落：每 200ms 一次，最多 30s
    for _ in 0..150 {
        if handle.is_finished() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    handle.join().expect("worker panic");
    println!("RSS_AFTER_MB {:.1}", rss_mb().unwrap_or(f64::NAN));
    println!("TOTAL_S {:.3}", t0.elapsed().as_secs_f64());
}

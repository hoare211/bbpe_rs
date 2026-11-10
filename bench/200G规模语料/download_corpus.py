#!/usr/bin/env python3
"""download_corpus.py — HF 成品语料的按量下载（对应《数据下载与预处理.md》§5）。

为什么不是 `huggingface-cli download` 一把梭：
  1. 大语料动辄几百 GB~3.8TB，全下浪费磁盘和带宽；本工具支持 --target-gb 到量即停；
  2. 需要按"单分片大小"决策下载策略（40GB 分片和 110MB 分片是两种问题）；
  3. 断点续传 + 完整性校验 + 磁盘预检，中断后重跑同一条命令即可继续。

实现方式：对文件清单走 HF 官方 `hf download --include <glob>`（保留其断点续传、
LFS/xet 支持），本脚本负责：文件清单盘点、按大小排序、配额切分、分批调用、汇总。

用法：
  # ① 盘点（不下载）
  python3 download_corpus.py list --repo opencsg/chinese-fineweb-edu-v2 --include "data/*.parquet"

  # ② 下载指定分片（glob）
  python3 download_corpus.py get --repo opencsg/chinese-fineweb-edu-v2 \
      --include "data/0000[01].parquet" --local-dir /data/raw/fineweb_edu

  # ③ 按量下载（到 20GB 即停；多来源各自独立）
  python3 download_corpus.py get --repo HuggingFaceFW/finewiki \
      --include "data/zhwiki/*.parquet" --local-dir /data/raw/finewiki --target-gb 6

  # ④ 限速下载（共享局域网时，最高 2MB/s）
  python3 download_corpus.py get --repo opencsg/chinese-fineweb-edu-v2 \
      --include "data/*.parquet" --local-dir /data/raw/fineweb_edu \
      --target-gb 200 --rate-limit-mb 2

限速说明：`hf download` 没有带宽参数，`--rate-limit-mb` 大于 0 时改走脚本内置的
令牌桶下载器（单线程串行、按 Range 续传、写盘进度可断点接续），速率上限按
MiB/s 计（2 ≈ 2.1 MB/s ≈ 17 Mbit/s）。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def hf_api(repo: str) -> dict:
    """调 HF API 拿文件清单（走 HF_ENDPOINT，默认官方；国内设 hf-mirror.com）。"""
    import urllib.request

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/api/datasets/{repo}"
    req = urllib.request.Request(url, headers={"User-Agent": "bbpe-download-tool/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001
        print(f"❌ HF API 请求失败：{url}\n   {e}", file=sys.stderr)
        print("   提示：国内设 export HF_ENDPOINT=https://hf-mirror.com", file=sys.stderr)
        raise SystemExit(2) from e


def collect_files(repo: str, include: str, exclude: list[str]) -> list[str]:
    data = hf_api(repo)
    siblings = [s["rfilename"] for s in data.get("siblings", [])]
    matched = [f for f in siblings if fnmatch.fnmatch(f, include)]
    for pat in exclude:
        matched = [f for f in matched if not fnmatch.fnmatch(f, pat)]
    # 排序保证"前 N 个"语义稳定（通常是 00000, 00001, ...）
    return sorted(matched)


def probe_size(repo: str, fname: str, attempts: int = 3) -> int | None:
    """HEAD 请求拿 X-Linked-Size（LFS 文件真实大小）。

    R4-24：失败返回 `None` 而不是 0。原实现把"探测失败"与"0 字节文件"
    混为一谈，于是配额累加 `todo_bytes += 0` 恒不增长，`--target-gb` 静默
    失效（会一路下载到磁盘满）。调用方必须显式区分这两种情况。

    2026-11-03：镜像对短时间内的密集 HEAD 也会回 429，一次失败就返回 None
    会让调用方误判为"探测能力全部失效"而整体退出，故内置 3 次退避重试。
    """
    import urllib.request

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{repo}/resolve/main/{fname}"
    for i in range(attempts):
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "bbpe-dl"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                size = int(r.headers.get("X-Linked-Size")
                           or r.headers.get("Content-Length") or 0)
                return size if size > 0 else None
        except Exception:  # noqa: BLE001
            if i + 1 < attempts:
                time.sleep(3 * (i + 1))
    return None


def existing_size(local_dir: Path, fname: str) -> int:
    p = local_dir / fname
    return p.stat().st_size if p.is_file() else 0


def probe_cache_path(repo: str) -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "bbpe-dl"
    return base / (repo.replace("/", "_") + ".sizes.json")


def load_probe_cache(repo: str) -> dict[str, int]:
    """探测结果缓存：HEAD 单次要 1~2s，625 个文件就是 10+ 分钟，重启时不该重付。

    2026-11-03 实测：把 150GB 拉起来前要探测 ~155 个分片，占 5 分钟；期间日志只剩
    一行"配额 150.0 GB"，看着像卡死。
    """
    p = probe_cache_path(repo)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in data.items() if int(v) > 0}
    except (OSError, ValueError, TypeError):
        return {}


def save_probe_cache(repo: str, cache: dict[str, int | None]) -> None:
    p = probe_cache_path(repo)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({k: v for k, v in cache.items() if v},
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)  # 原子替换：多分片并发写同一缓存也不会写坏
    except OSError:
        pass


def resolve_hf_cli() -> str:
    """定位 `hf` 可执行文件：优先 PATH，其次当前解释器同目录（uv venv 未激活时）。"""
    found = shutil.which("hf")
    if found:
        return found
    candidate = Path(sys.executable).parent / "hf"
    return str(candidate) if candidate.is_file() else "hf"


def download_throttled(repo: str, fname: str, local_dir: Path,
                       rate_mbps: float, expected: int | None,
                       rate_file: Path | None = None,
                       rate_divisor: float = 1.0) -> int:
    """令牌桶限速下载单个文件（支持 Range 续传），返回本次新增字节数。

    只做 HTTP 直连下载，不经过 hf CLI —— 正因为 hf/xet 客户端全部内置多线程且
    无法限速，才需要这条"单流 + 睡眠补偿"的通路。

    rate_file 存在时每 5 秒重读一次（单位 MiB/s），改速率不必重启下载：
      echo 10 > output/.rate_mb
    """
    import urllib.error
    import urllib.request

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{repo}/resolve/main/{fname}"
    dest = local_dir / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.is_file() else 0

    headers = {"User-Agent": "bbpe-dl"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # 本地已完整
            return 0
        raise

    # 服务端忽略 Range 时（返回 200）必须从头覆盖，否则文件会被写脏
    resume = have > 0 and resp.status == 206
    if not resume:
        have = 0
    mode = "ab" if resume else "wb"

    rate = rate_mbps * 1024 * 1024  # MiB/s → 字节/s
    chunk = 256 * 1024  # 读块大一点：64KB 时每次要睡 ~120ms，读/睡比太低
    added = 0          # 本会话累计（返回值）
    win = 0            # 当前限速窗口内字节数（速率变更时归零重算）
    t_start = time.monotonic()
    t_win = t_start
    t_conn = t_start   # 本条连接的建立时间（用于老化重连）
    last_report = t_start
    last_added = 0
    last_rate_check = 0.0
    with dest.open(mode) as f:
        while True:
            buf = resp.read(chunk)  # 读超时（socket timeout=120s）交由调用方重试
            if not buf:
                break
            f.write(buf)
            added += len(buf)
            win += len(buf)
            now = time.monotonic()

            # 热更新速率：只改文件，不必重启进程
            if rate_file and now - last_rate_check >= 5:
                last_rate_check = now
                try:
                    # 文件里写的是**总**配额；多流并行时每流只分到 1/divisor
                    v = float(rate_file.read_text(encoding="utf-8").strip()) / rate_divisor
                except (OSError, ValueError):  # 文件缺失/半写入时沿用旧值
                    v = 0
                if v > 0 and abs(v * 1024 * 1024 - rate) > 1024:
                    rate = v * 1024 * 1024
                    win = 0                  # 归零，避免按新速率一次性补发积压
                    t_win = now
                    print(f"      限速调整为 {v:g} MiB/s（即时生效）", flush=True)

            # 睡眠补偿：窗口内这些字节"应该"耗时 win/rate，落后多少就睡多少
            behind = win / rate - (now - t_win)
            if behind > 0:
                time.sleep(min(behind, 1.0))

            if now - last_report >= 30:
                recent = (added - last_added) / max(now - last_report, 1e-6)
                last_report = now
                last_added = added
                speed = added / max(now - t_start, 1e-6) / 1e6
                print(f"      {fname}: {added/1e6:.0f} MB @ {speed:.2f} MB/s"
                      f"（近 30s {recent/1e6:.2f} MB/s）", flush=True)
                # 连接质量判定：只在"几乎不动"时换连接。
                # 2026-11-03 实测教训：阈值设成 0.5×配额 会反复掐掉正在爬坡的
                # 连接（新连接前 1~2 分钟不稳定），4 条流互相拖累，聚合吞吐反而
                # 掉到 0.65 MiB/s；而同刻 4 条 curl 能跑满 4 MB/s。
                if now - t_conn > 120 and recent < 0.1 * rate:
                    resp.close()
                    raise Reconnect(f"近 30s 仅 {recent/1e6:.2f} MB/s（配额 {rate/1e6:.1f}）")
                if now - t_conn > 900:
                    resp.close()
                    raise Reconnect(f"连接已用 {int(now - t_conn)}s")
    resp.close()

    size = dest.stat().st_size
    if expected is not None and size != expected:
        raise IOError(f"{fname}: 大小不符（期望 {expected}，实际 {size}）")
    return added


def download_with_curl(repo: str, fname: str, local_dir: Path,
                       rate_mbps: float, expected: int | None) -> int:
    """用 curl --limit-rate 传一个文件（-C - 断点续传），返回本会话新增字节数。

    为什么不用内置读取器：应用层"读一块睡一会儿"会让 TCP 长期处于 app-limited，
    好连接也被拖成 0.2~0.3 MB/s；curl 的限速实现能在好连接上精确跑满配额
    （2026-11-03 实测 500 KB/s 档位稳定 512 KB/s，4 条并发聚合 1.6 MB/s）。
    """
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("找不到 curl，改用 --engine builtin")

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{repo}/resolve/main/{fname}"
    dest = local_dir / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    before = dest.stat().st_size if dest.is_file() else 0
    if expected is not None and before == expected:
        return 0

    cmd = [curl, "-L", "-sS", "--fail", "-C", "-", "-o", str(dest)]
    if rate_mbps and rate_mbps > 0:  # ≤0 表示不限速：直接全带宽拉
        cmd += ["--limit-rate", f"{rate_mbps * 1024:.0f}k"]
    cmd += ["--connect-timeout", "30", "--retry", "5", "--retry-delay", "5",
            "--retry-all-errors", "--max-time", "7200", url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    after = dest.stat().st_size if dest.is_file() else 0
    if r.returncode != 0:
        msg = (r.stderr or "").strip()
        msg = msg.splitlines()[-1] if msg else f"curl 退出码 {r.returncode}"
        # 416 / "already fully downloaded"：本地已经是完整文件，等价于跳过
        if r.returncode == 33 or "416" in msg or "already fully" in msg.lower():
            return 0
        raise IOError(msg)
    if expected is not None and after != expected:
        raise IOError(f"{fname}: 大小不符（期望 {expected}，实际 {after}）")
    return max(after - before, 0)


class Reconnect(Exception):
    """连接已老化/被 CDN 限速：立刻换一条新连接续传，不算失败。

    2026-11-03 实测：同一条连接跑久了会被 cas-bridge 压到 ~90 KB/s，而同一时刻
    新建连接能跑到 2.9 MB/s（远超我们 2 MiB/s 的配额）⟹ 直接重连是最省事的解法。
    """


def is_retryable(exc: Exception) -> bool:
    """瞬时错误判定：限流/网关错误/连接与读超时都重试，403/404 这类直接放弃。

    2026-11-03 实测：hf-mirror 长时间单流下载会返回 429（Too Many Requests），
    另有连接假死导致的读超时；两者都是"等一下再续"就能过的情况。
    """
    import socket
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 425, 429, 500, 502, 503, 504)
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionError,
                        urllib.error.URLError)):
        return True
    return False


def retry_delay(exc: Exception, attempt: int, base: float = 30.0) -> float:
    """重试等待：优先服务端 Retry-After，否则指数退避（30s→60s→…，上限 10 分钟）。"""
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError) and exc.headers:
        ra = exc.headers.get("Retry-After")
        if ra:
            try:
                return max(float(ra), 5.0)
            except ValueError:
                pass
    return min(base * (2 ** (attempt - 1)), 600.0)


def cmd_list(args) -> int:
    files = collect_files(args.repo, args.include, args.exclude or [])
    if not files:
        print(f"❌ 没有匹配 {args.include!r} 的文件", file=sys.stderr)
        return 1
    print(f"== HF 文件清单：{args.repo} ==")
    print(f"  匹配 {args.include!r} ⟹ {len(files)} 个文件")

    # 只探测前若干个 + 末若干个，避免大量 HEAD 请求
    probe_n = min(args.probe, len(files))
    sizes = {}
    for f in files[:probe_n]:
        sizes[f] = probe_size(args.repo, f)
    known = [s for s in sizes.values() if s]
    if known:
        avg = sum(known) / len(known)
        est_total_gb = avg * len(files) / 1e9
        print(f"  单分片大小（探测前 {probe_n} 个）："
              f"min {min(known)/1e6:.0f}MB / avg {avg/1e6:.0f}MB / max {max(known)/1e6:.0f}MB")
        print(f"  估算总量：~{est_total_gb:.1f} GB（按均值推算，仅供参考）")
        print()
        print("  💡 下载量决策（峰值 RSS ≈ 9.2 × 训练语料字节数）：")
        for rate in (0.02, 0.05, 0.10):
            need = est_total_gb * rate
            print(f"     采样率 {rate:.0%} ⟹ 需下载 ~{need:.1f} GB（训练语料）"
                  f" ⟹ 峰值 ~{9.2*need:.0f} GB 内存")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"repo": args.repo, "include": args.include, "files": files,
             "probed_sizes_bytes": sizes}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  清单已存: {args.json}")
    return 0


def cmd_get(args) -> int:
    files = collect_files(args.repo, args.include, args.exclude or [])
    if not files:
        print(f"❌ 没有匹配 {args.include!r} 的文件", file=sys.stderr)
        return 1

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # 磁盘预检
    need_free = (args.target_gb or 0) * 1e9 * 1.5  # 留 50% 余量给后续转换产物
    free = shutil.disk_usage(local_dir).free
    if need_free and free < need_free:
        print(f"❌ 磁盘空间不足：需要 ~{need_free/1e9:.1f}GB，{local_dir} 仅剩 {free/1e9:.1f}GB",
              file=sys.stderr)
        return 1

    # 探测大小 + 断点续传：已存在且大小一致的跳过
    print(f"== 下载：{args.repo} ==")
    print(f"  匹配 {len(files)} 个文件 | 目标目录 {local_dir}")
    if args.target_gb:
        print(f"  配额：{args.target_gb} GB（到量即停）")

    todo: list[str] = []
    todo_sizes: dict[str, int | None] = {}
    # 配额以"目录最终体积"为准：已落盘的部分先计入，避免重复计算导致超额。
    # 2026-11-03 实测：旧口径只累加计划量、不扣已在盘上的量，wudao 目标 15GB
    # 实际下到 17.2GB 还在继续。
    done_bytes = sum(existing_size(local_dir, f) for f in files)
    todo_bytes = done_bytes
    skipped = 0
    target_bytes = (args.target_gb or float("inf")) * 1e9

    # R4-24：先探测一批（≤64 个）作为"未知大小"文件的估算基准。
    # 配额已启用却**一个都探测失败**时必须直接失败：否则 `todo_bytes` 恒为 0，
    # `--target-gb` 静默退化成"一直下到磁盘满"。
    cached = load_probe_cache(args.repo)
    size_cache: dict[str, int | None] = {f: cached[f] for f in files if f in cached}
    if cached:
        print(f"  复用探测缓存 {sum(1 for f in files if f in cached)} 个分片（免 HEAD 往返）")
    # 只探测 8 个作基准：密集 HEAD 会被镜像限流（429），探测数越多越容易全灭
    for f in [x for x in files if x not in size_cache][:8]:
        size_cache[f] = probe_size(args.repo, f)
        time.sleep(0.4)
    known_sizes = [s for s in size_cache.values() if s]
    avg_size = sum(known_sizes) / len(known_sizes) if known_sizes else None
    if args.target_gb and avg_size is None:
        print("  ⚠️ HEAD 全部失败，60s 后重试一次…", file=sys.stderr)
        time.sleep(60)
        for f in [x for x in files if x not in size_cache][:3]:
            size_cache[f] = probe_size(args.repo, f)
            time.sleep(2)
        known_sizes = [s for s in size_cache.values() if s]
        avg_size = sum(known_sizes) / len(known_sizes) if known_sizes else None
    if args.target_gb and avg_size is None:
        print(
            "❌ 无法探测任何文件大小（HEAD 请求全部失败），--target-gb 无法生效。",
            file=sys.stderr,
        )
        print(
            "   常见原因：镜像限流（429）/ 镜像不支持 HEAD / 代理拦截 / 需登录的 gated 仓库。\n"
            "   处置：等几分钟再跑（本脚本已内置重试），或改用 --include 精确到少量分片后再下。",
            file=sys.stderr,
        )
        return 1

    unknown = 0
    for f in files:
        # 只用手上的缓存/基准探测值，**不再逐文件 HEAD**：625 个分片逐个探测要
        # 10+ 分钟（单次 1~2s），期间日志不动、看着像卡死，还容易被镜像 429。
        # 未知大小一律按均值估算配额；下载器拿到 expected=None 就不做大小校验，
        # 已完整的文件会因 Range 416 直接返回 0 字节，等价于"跳过"。
        want = size_cache.get(f)
        have = existing_size(local_dir, f)
        if want is not None and have == want:
            skipped += 1          # 已完整：其体积已包含在 done_bytes 里
            continue
        if todo_bytes >= target_bytes:
            break
        todo.append(f)
        todo_sizes[f] = want
        if want is not None:
            todo_bytes += max(want - have, 0)   # 只补差量，续传不重复计
        else:
            # 大小未知：按已探测均值估算增量（宁可少下，也不要突破 --target-gb）
            unknown += 1
            todo_bytes += max(int(avg_size or 0) - have, 0)

    save_probe_cache(args.repo, size_cache)

    # 多流并行：配额与待办列表按**全局**算好，再按索引取模切给第 K 条流。
    # 早先按"每流 target/N"分配，会因为单个分片就超过 N 分之一配额而让
    # 大多数流误判"配额已满"直接退出（2026-11-03 实测 6 条流只剩 1 条在动）。
    if getattr(args, "shard", None):
        k_s, n_s = args.shard.split("/")
        k, n = int(k_s), int(n_s)
        if not 0 <= k < n:
            print(f"❌ --shard {args.shard} 不合法（应为 K/N，0 ≤ K < N）", file=sys.stderr)
            return 1
        todo = [f for i, f in enumerate(todo) if i % n == k]
        if not todo:
            print(f"  ⏭ 第 {k}/{n} 条流没分到待下文件，任务结束")
            return 0

    print(f"  已在盘 {done_bytes/1e9:.2f} GB（完整 {skipped} 个）| 待下载 {len(todo)} 个 "
          f"⟹ 完成后约 {todo_bytes/1e9:.2f} GB")
    if unknown:
        print(
            f"  ⚠️ {unknown} 个分片未逐个探测，配额按均值 "
            f"~{(avg_size or 0)/1e6:.0f}MB 估算（实际以磁盘为准）",
            file=sys.stderr,
        )
    if not todo:
        print("  ✅ 配额内全部就绪，无需下载")
        return 0

    # 分批调用 hf download（官方 CLI 负责续传/LFS），单批 ≤ 64 个文件
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    env = dict(os.environ, HF_ENDPOINT=endpoint)

    # 限速模式：hf/xet 客户端无法限速，改走内置限速通路。
    # --engine curl 且 --rate-limit-mb 0 ⟹ 不限速的 curl 多流（补数据时用满带宽）。
    if (args.rate_limit_mb and args.rate_limit_mb > 0) or args.engine == "curl":
        remaining = 0
        for f in todo:
            want = todo_sizes.get(f)
            remaining += ((want or 0) - existing_size(local_dir, f)) if want else int(avg_size or 0)
        engine = args.engine
        if engine == "auto":
            engine = "curl" if shutil.which("curl") else "builtin"
        if args.rate_limit_mb and args.rate_limit_mb > 0:
            eta_h = remaining / (args.rate_limit_mb * 1024 * 1024) / 3600
            print(f"  限速 {args.rate_limit_mb:g} MiB/s（引擎 {engine}）⟹ "
                  f"待传输 ~{remaining/1e9:.2f} GB，预计 {eta_h:.1f} 小时")
        else:
            print(f"  不限速（引擎 {engine}）⟹ 待传输 ~{remaining/1e9:.2f} GB")
        total_added = 0
        t0 = time.monotonic()
        for idx, f in enumerate(todo, 1):
            print(f"\n  [{idx}/{len(todo)}] {f}")
            attempt = 0
            reconnects = 0
            # 每文件重读一次热更新限速（curl 的 --limit-rate 是启动参数，只能逐文件生效）
            rate_now = args.rate_limit_mb
            if args.rate_file:
                try:
                    v = float(Path(args.rate_file).read_text(encoding="utf-8").strip())
                    if v > 0:
                        rate_now = v / args.rate_divisor
                except (OSError, ValueError):
                    pass
            while True:
                attempt += 1
                try:
                    if engine == "curl":
                        added = download_with_curl(
                            args.repo, f, local_dir, rate_now, todo_sizes.get(f))
                    else:
                        added = download_throttled(
                            args.repo, f, local_dir, rate_now, todo_sizes.get(f),
                            Path(args.rate_file) if args.rate_file else None,
                            args.rate_divisor)
                    break
                except KeyboardInterrupt:
                    print("\n  ⏸ 已中断。重跑本命令将从断点续传。", file=sys.stderr)
                    return 130
                except Reconnect as e:
                    # 只换连接不换目标：进度已落盘，下一次 Range 继上；不计入失败次数
                    reconnects += 1
                    attempt -= 1
                    print(f"      ↻ {e}；换新连接续传（第 {reconnects} 次）",
                          file=sys.stderr, flush=True)
                    if reconnects > 300:
                        print("  ⚠️ 重连次数过多，放弃本文件", file=sys.stderr)
                        return 1
                    time.sleep(3)
                    continue
                except Exception as e:  # noqa: BLE001
                    # 429/超时是限速镜像的常态，退避后续传，别把整条链带崩
                    if attempt <= args.retries and is_retryable(e):
                        wait = retry_delay(e, attempt)
                        print(f"      ⚠️ 第 {attempt}/{args.retries} 次失败：{e}；"
                              f"{wait:.0f}s 后从断点重试", file=sys.stderr, flush=True)
                        time.sleep(wait)
                        continue
                    print(f"  ⚠️ 失败：{e}（重跑本命令自动续传）", file=sys.stderr)
                    return 1
            total_added += added
            # 落盘即记账：把下载完成后的真实大小写回探测缓存，下次重跑就能
            # 一眼跳过已完整分片，不必再发 Range 试探（366 个文件≈6 分钟）
            final = existing_size(local_dir, f)
            if final > 0:
                todo_sizes[f] = final
                size_cache[f] = final
                if idx % 5 == 0:
                    save_probe_cache(args.repo, size_cache)
            print(f"      ✅ 本次新增 {added/1e6:.1f} MB")
        save_probe_cache(args.repo, size_cache)
        dur = time.monotonic() - t0
        print(f"\n  ✅ 下载完成：{local_dir}")
        print(f"     新增 {total_added/1e9:.2f} GB | 耗时 {dur/3600:.2f} 小时 | "
              f"平均 {total_added/max(dur,1)/1e6:.2f} MB/s")
        return 0

    batch = 64
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        cmd = [resolve_hf_cli(), "download", "--repo-type", "dataset", args.repo, *chunk,
               "--local-dir", str(local_dir), "--max-workers", str(args.max_workers)]
        print(f"\n  [{i//batch + 1}] hf download …（{len(chunk)} 个文件）")
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"\n  ⚠️ 批次失败（退出码 {r.returncode}）。重跑本命令将自动跳过已完成部分。",
                  file=sys.stderr)
            return r.returncode
    print(f"\n  ✅ 下载完成：{local_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="HF 语料按量下载（list / get）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", required=True, help="如 opencsg/chinese-fineweb-edu-v2")
    common.add_argument("--include", required=True, help='glob，如 "data/*.parquet"')
    common.add_argument("--exclude", action="append", default=[], help="额外排除的 glob")

    p_list = sub.add_parser("list", parents=[common], help="只盘点文件清单（不下载）")
    p_list.add_argument("--probe", type=int, default=8, help="探测前 N 个文件大小")
    p_list.add_argument("--json", default=None, help="清单输出路径")
    p_list.set_defaults(fn=cmd_list)

    p_get = sub.add_parser("get", parents=[common], help="下载（支持配额/续传）")
    p_get.add_argument("--local-dir", required=True)
    p_get.add_argument("--target-gb", type=float, default=None, help="到量即停（GB）")
    p_get.add_argument("--max-workers", type=int, default=4)
    p_get.add_argument("--rate-limit-mb", type=float, default=0,
                       help="限速上限（MiB/s），>0 时改走内置单流限速下载器；0=不限速")
    p_get.add_argument("--retries", type=int, default=10,
                       help="单文件瞬时错误（429/超时/5xx）的重试次数，指数退避")
    p_get.add_argument("--rate-file", default=None,
                       help="限速热更新文件（单位 MiB/s，含总配额），运行中改它即可变速")
    p_get.add_argument("--rate-divisor", type=float, default=1.0,
                       help="多流并行时每流只取 rate-file 值的 1/N（N 为并行流数）")
    p_get.add_argument("--shard", default=None, metavar="K/N",
                       help="多流并行：只下第 K 份（0 起）文件，如 0/3、1/3")
    p_get.add_argument("--engine", default="auto", choices=["auto", "curl", "builtin"],
                       help="限速下载引擎：curl（--limit-rate，推荐）/ builtin / auto")
    p_get.set_defaults(fn=cmd_get)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

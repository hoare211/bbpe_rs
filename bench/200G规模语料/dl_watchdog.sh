#!/usr/bin/env bash
# dl_watchdog.sh — 200GB 语料下载链的定时看护（对应《数据下载与预处理.md》§5.3）
#
# 每 INTERVAL 秒做一次巡检，动作分三种：
#   1. 链日志出现 "ALL DONE" ⟹ 退出；
#   2. 找不到 download_corpus.py 进程 ⟹ 重新拉起整条链（每个源幂等：已下的分片跳过）；
#   3. 进程在但 raw/ 总字节数连续两轮不增长（网络假死）⟹ 杀掉后重拉，靠 Range 断点续传。
#
# 用法：
#   nohup setsid bash bench/200G规模语料/dl_watchdog.sh >/dev/null 2>&1 &
#   tail -f /data-hdd/home2/hoare/bbpe_rs_bench/output/dl_watchdog.log
set -uo pipefail

BASE="${BASE:-/data-hdd/home2/hoare/bbpe_rs_bench}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv-bench/bin/python"
DL="$ROOT/bench/200G规模语料/download_corpus.py"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

INTERVAL="${INTERVAL:-1800}"      # 每 30 分钟一次
RATE="${RATE:-2}"                 # **总**限速 MiB/s（各并行流加起来不超过它）
SHARDS="${SHARDS:-6}"             # 每个源的并行流数（单流易被 CDN 压速）

OUT="$BASE/output"
LOG="$OUT/dl_watchdog.log"
CHAIN_LOG="$OUT/dl_chain.log"
STATE="$OUT/.dl_watchdog_state"
LOCK="$OUT/.dl_watchdog.lock"
PIDFILE="$OUT/.dl_chain.pid"
mkdir -p "$OUT"

# 单实例：避免重复拉起两个 watchdog 各自重启下载
exec 9>"$LOCK"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
  echo "已有 watchdog 在跑（$LOCK 被占用），退出" >&2
  exit 0
fi

log() { echo "$(date +%F_%T) $*" >> "$LOG"; }

# 整条链：顺序固定，每步各自独立配额；重跑时会跳过已完整的分片
start_chain() {
  # 上一轮可能因瞬时错误提前收尾而留下 ALL DONE 标记 ⟹ 归档，否则 watchdog 立刻退出
  if grep -q "ALL DONE" "$CHAIN_LOG" 2>/dev/null; then
    mv "$CHAIN_LOG" "${CHAIN_LOG%.log}.$(date +%F_%H%M%S).log"
  fi
  # 限速热更新文件：运行中 echo <新速率MiB/s> > 它即可变速，不用重启
  [ -f "$OUT/.rate_mb" ] || echo "$RATE" > "$OUT/.rate_mb"
  setsid bash -c "
set -u
dl() {
  local r pids=() k rc
  echo \"=== \$(date +%F_%T) START \$3 (target \$4GB x $SHARDS 流) ===\" >> \"$CHAIN_LOG\"
  # 配额按全局给（各流自己算同一份 todo，只下自己那一片），速率按总量除以流数
  r=\$(awk -v t=\"$RATE\" -v n=\"$SHARDS\" 'BEGIN{printf \"%.3f\", t/n}')
  for ((k=0; k<$SHARDS; k++)); do
    \"$PY\" \"$DL\" get --repo \"\$1\" --include \"\$2\" --local-dir \"$BASE/raw/\$3\" \\
        --target-gb \"\$4\" --rate-limit-mb \"\$r\" --engine curl \\
        --rate-file \"$OUT/.rate_mb\" --rate-divisor $SHARDS --shard \"\$k/$SHARDS\" \\
        >> \"$OUT/dl_\$3.shard\$k.log\" 2>&1 &
    pids+=(\$!)
  done
  wait \"\${pids[@]}\"
  rc=\$?   # 必须先存下来：紧接的 \$(date) 展开会把 \$? 覆盖成 0
  echo \"--- \$(date +%F_%T) END \$3 exit=\$rc ---\" >> \"$CHAIN_LOG\"
}
dl HuggingFaceFW/finewiki         \"data/zhwiki/*\"    finewiki     5.5
dl p208p2002/wudao                \"*.parquet\"        wudao        15
dl Skywork/SkyPile-150B           \"data/*.jsonl\"     skypile      30
dl opencsg/chinese-fineweb-edu-v2 \"data/*.parquet\"   fineweb_edu  150
echo \"=== \$(date +%F_%T) ALL DONE ===\" >> \"$CHAIN_LOG\"
" 9>&- > /dev/null 2>&1 < /dev/null &
  echo "$!" > "$PIDFILE"   # setsid 后 bash 自身即新进程组的组长，pid == pgid
}

# 停掉整条链：优先按进程组杀（连子进程一起），兜底再按命令行匹配
stop_chain() {
  if [ -f "$PIDFILE" ]; then
    local pgid; pgid="$(cat "$PIDFILE" 2>/dev/null)"
    [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null
    rm -f "$PIDFILE"
  fi
  pkill -f "download_corpus.py get" 2>/dev/null
  sleep 3
  pkill -9 -f "download_corpus.py get" 2>/dev/null
  return 0
}

raw_bytes() { du -sb "$BASE/raw" 2>/dev/null | cut -f1; }

log "watchdog 启动 | interval=${INTERVAL}s rate=${RATE}MiB/s root=$ROOT"
prev=""; stall=0

while true; do
  if grep -q "ALL DONE" "$CHAIN_LOG" 2>/dev/null; then
    log "链日志显示 ALL DONE，watchdog 退出 | raw=$(numfmt --to=iec "$(raw_bytes)")"
    exit 0
  fi

  cur="$(raw_bytes)"
  if pgrep -f "download_corpus.py get" >/dev/null 2>&1; then
    log "运行中 | raw=$((cur/1000000)) MB | 当前源=$(grep -E 'START|END' "$CHAIN_LOG" 2>/dev/null | tail -1 | sed 's/===//g')"
    if [ -n "$prev" ] && [ "$cur" = "$prev" ]; then
      stall=$((stall + 1))
      log "⚠️ 本轮无增长（连续 $stall 次）"
      if [ "$stall" -ge 2 ]; then
        log "⚠️ 判定假死，杀掉下载进程并重拉链"
        stop_chain
        start_chain
        stall=0
      fi
    else
      stall=0
    fi
  else
    log "未检测到下载进程（raw=$((cur/1000000)) MB），重拉链"
    start_chain
    sleep 20
    pgrep -f "download_corpus.py get" >/dev/null 2>&1 \
      && log "重拉成功" || log "❌ 重拉后仍未起来，检查 $OUT/dl_*.log"
    stall=0
  fi

  prev="$cur"
  # 9>&- 必须带：否则长睡的子进程会继承锁 fd，watchdog 被杀后锁仍被占，
  # 之后重启会误判"已有 watchdog 在跑"而永久退出（2026-11-03 实测踩到）
  sleep "$INTERVAL" 9>&-
done

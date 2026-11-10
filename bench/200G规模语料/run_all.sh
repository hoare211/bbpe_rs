#!/usr/bin/env bash
# run_all.sh — 200G 语料分词器训练端到端流水线（对应 README §6）
#
# 用法：
#   bash bench/200G规模语料/run_all.sh /data/corpus/train.jsonl 150256
#
# 步骤：prepare(可选跳过) → train(2% 采样 + held-out) → inspect → sweep →
#       compare(5% 嵌套样本) → eval_quality → bench_encode → compare_engines(三方对照)
# 全部日志与产物落在 bench/200G规模语料/output/
set -euo pipefail

CORPUS="${1:?用法: run_all.sh <train.jsonl> <总词表 如 150256>}"
VOCAB="${2:-150256}"
SAMPLE_RATE="${SAMPLE_RATE:-0.02}"
HELDOUT_RATE="${HELDOUT_RATE:-0.005}"
COMPARE_RATE="${COMPARE_RATE:-0.05}"

BBPE="${BBPE:-target/release/bbpe}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/output"
mkdir -p "$OUT"

log() { echo -e "\n\033[1;36m== $* ==\033[0m" | tee -a "$OUT/pipeline_main.log"; }

cd "$ROOT"

log "0/7 环境检查"
command -v "$BBPE" >/dev/null || { echo "❌ 未找到 $BBPE，先 cargo build --release --no-default-features"; exit 1; }
# R4-10：标准安装（pyproject.toml: module-name = "bbpe_tokenizer.bbpe_rs"）下
# 顶层 `import bbpe_rs` 不存在，实测路径是 `bbpe_tokenizer.bbpe_rs`。
# 原脚本只试顶层路径，失败仅打印"跳过"，质量/性能两步被静默略过而流水线
# 仍报"完成"。现在：按候选路径探测，**全部失败即阻断**（这两步是本流水线
# 的组成部分，不是可选装饰）。
if PYTHONPATH="$ROOT/python" python3 -c "
import importlib, sys
for name in ('bbpe_tokenizer.bbpe_rs', 'bbpe_rs'):
    try:
        importlib.import_module(name); sys.exit(0)
    except ImportError:
        continue
sys.exit(1)
" 2>/dev/null; then
    echo "✅ Python 绑定可用（bbpe_tokenizer.bbpe_rs 或 bbpe_rs）"
else
    echo "❌ 找不到 Rust Python 绑定（已尝试 bbpe_tokenizer.bbpe_rs / bbpe_rs）。" >&2
    echo "   先执行：maturin develop --release --skip-install" >&2
    echo "   （pyproject.toml 声明 module-name = \"bbpe_tokenizer.bbpe_rs\"，装在包内）" >&2
    exit 1
fi
nproc; free -g | head -2

log "1/7 训练（采样 ${SAMPLE_RATE} + held-out ${HELDOUT_RATE}）"
# shellcheck disable=SC2086
time "$BBPE" train \
    --input "$CORPUS" --format jsonl \
    --vocab "$VOCAB" \
    --output output/vocab_${VOCAB}.json \
    --sample-rate "$SAMPLE_RATE" --sample-seed 42 \
    --heldout-rate "$HELDOUT_RATE" --heldout-seed 7 \
    --heldout-out data/heldout.jsonl \
    --sample-out data/train_sample.jsonl \
    2>&1 | tee "$OUT/train.log"

log "2/7 模型契约校验（inspect V1–V7）"
"$BBPE" inspect "output/vocab_${VOCAB}.json" --validate-only | tee "$OUT/inspect.log"

log "3/7 sweep 选 K 并导出（一次训练，贪心前缀性质）"
"$BBPE" sweep --model "output/vocab_${VOCAB}.json" --held-out data/heldout.jsonl \
    --vocabs 33024,49408,65792,98560,131328,"$VOCAB" \
    --export-at 100256 --max-bytes 52428800 \
    2>&1 | tee "$OUT/sweep.log"

log "4/7 采样充分性：${COMPARE_RATE} 嵌套样本对照（分叉点判据）"
# R4-12：对照训练必须带**与基准训练相同**的 held-out 参数（rate/seed 一致）。
# 采样器会把 held-out 命中行从训练集排除；两份训练输入的 held-out 配置不同
# 时，train_2% ⊇ train_5% 的嵌套性被破坏，compare 的结果混入了 held-out
# 差异而非采样率差异。held-out 输出写到单独文件（确定性采样下内容与基准
# 训练产出的 data/heldout.jsonl 相同，不覆盖它以免影响 sweep 已用过的文件）。
# shellcheck disable=SC2086
"$BBPE" train --input "$CORPUS" --format jsonl --vocab "$VOCAB" \
    --output "output/vocab_${VOCAB}_$(echo "$COMPARE_RATE" | tr -d .).json" \
    --sample-rate "$COMPARE_RATE" --sample-seed 42 \
    --heldout-rate "$HELDOUT_RATE" --heldout-seed 7 \
    --heldout-out data/heldout_compare.jsonl \
    2>&1 | tee "$OUT/train_compare.log"
"$BBPE" compare \
    --a "output/vocab_${VOCAB}_$(echo "$COMPARE_RATE" | tr -d .).json" \
    --b "output/vocab_${VOCAB}.json" \
    --at 1000,10000,50000,100000,"$((VOCAB - 256))" \
    2>&1 | tee "$OUT/compare.log"

# R4-11：sweep 的导出文件名是 `<源模型 stem>_<总词表>.json`（sweep.rs），
# 即 output/vocab_${VOCAB}_100256.json —— 原脚本固定评估
# output/vocab_100256.json，导入问题修好后仍会因文件不存在失败；
# 若该路径存在旧文件，还会**静默评估错误的模型**。现在与导出命名对齐，
# 并在评估前显式校验文件存在（缺失即失败，不静默跳过）。
EXPORT_AT=100256
MODEL="output/vocab_${VOCAB}_${EXPORT_AT}.json"
if [ ! -f "$MODEL" ]; then
    echo "❌ 未找到 sweep 导出模型 $MODEL（sweep --export-at ${EXPORT_AT} 应产出该文件）" >&2
    echo "   若改名失败，请检查上面的 sweep 日志" >&2
    exit 1
fi

log "5/7 正确性 + 质量（round-trip / offsets / special / fertility）"
# R4-10：导入探测已在 0/6 阻断，这里直接跑（失败由 set -e 传播）
PYTHONPATH="$ROOT/python" python3 "$HERE/eval_quality.py" --model "$MODEL" \
    --input data/heldout.jsonl --format jsonl \
    --json "$OUT/quality_report.json" | tee "$OUT/quality.log"

log "6/7 编解码性能（吞吐 / 加速比 / 延迟 / RSS）"
PYTHONPATH="$ROOT/python" python3 "$HERE/bench_encode.py" --model "$MODEL" \
    --input data/heldout.jsonl --format jsonl --threads 1,4,8,16 \
    --json "$OUT/bench_encode.json" | tee "$OUT/bench_encode.log"

log "7/7 三方编码对照（bbpe / tiktoken / HF tokenizers，同文本同进程数）"
# 说明（2026-11-07）：
# - 对照必须**同一批次**测完，绝对吞吐会随机器状态漂移（实测同配置不同时刻差 20%），
#   跨批次只能比比值；
# - HF 训练内存约 67×语料（1GB 语料峰值 ~68GB），**无法吃全量语料**，
#   故先用 HF_MAX_GB 切片，默认 1GB；SKIP_HF=1 可跳过 HF 一栏；
# - 依赖装在 .venv-bench（uv），系统 python3 通常没有 tokenizers/tiktoken。
PY_BENCH="${PY_BENCH:-$ROOT/.venv-bench/bin/python}"
[ -x "$PY_BENCH" ] || PY_BENCH="python3"
HF_MODEL="output/hf_bpe_${VOCAB}.json"
HF_ARGS=()
if [ "${SKIP_HF:-0}" = "1" ]; then
    echo "  ⚠️ SKIP_HF=1：跳过 HF 训练与对照，只比 bbpe / tiktoken"
elif [ -f "$HF_MODEL" ]; then
    echo "  复用已有 HF 模型：$HF_MODEL"
    HF_ARGS=(--hf-model "$HF_MODEL")
else
    HF_CORPUS="${HF_CORPUS:-data/train_sample.jsonl}"
    HF_MAX_GB="${HF_MAX_GB:-1}"
    if [ ! -f "$HF_CORPUS" ]; then
        echo "  ⚠️ 找不到 $HF_CORPUS（第 1 步未产出采样文件），跳过 HF"
    else
        echo "  HF 未就绪：按 ${HF_MAX_GB}GB 切片训练（内存约 67×语料，不能吃全量）"
        RAYON_NUM_THREADS="${HF_THREADS:-8}" \
        taskset -c "${HF_CORES:-0-7}" "$PY_BENCH" "$HERE/bench_hf.py" train \
            "$HF_CORPUS" "$VOCAB" "$HF_MODEL" "$HF_MAX_GB" \
            2>&1 | tee "$OUT/hf_train.log"
        [ -f "$HF_MODEL" ] && HF_ARGS=(--hf-model "$HF_MODEL")
    fi
fi

# shellcheck disable=SC2086
PYTHONPATH="$ROOT/python" "$PY_BENCH" "$HERE/compare_engines.py" \
    --from-jsonl data/heldout.jsonl --max-gb "${COMPARE_GB:-1}" \
    --text-out "$OUT/compare_text.txt" \
    --bbpe-model "$MODEL" "${HF_ARGS[@]}" \
    --procs "${COMPARE_PROCS:-8}" \
    --json "$OUT/compare_engines.json" 2>&1 | tee "$OUT/compare_engines.log"

log "完成。产物：$OUT/（train/sweep/compare/quality/bench_encode/hf_train/compare_engines 日志与 json）"
echo "  最终模型: $MODEL（K_max 全量: output/vocab_${VOCAB}.json）"

use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

// R6-1 P2：程序化取消 / 超时使用独立异常类型，便于用户与 KeyboardInterrupt 区分。
// 信号路径仍原样返回 signal handler 抛出的 PyErr。
create_exception!(bbpe_tokenizer.bbpe_rs, TrainingCancelled, PyRuntimeError);
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::cancel::{CancelReason, CancelToken};
use crate::cli::util::{
    PEAK_MEM_FACTOR, TempFileGuard, available_ram_bytes, ensure_parent, exceeds_memory_budget,
    human_bytes, same_path, temp_path,
};
use crate::compare::{CompareReport, DEFAULT_AT, compare_states};
use crate::error::BbpeError;
use crate::pre_tokenizer::PreTokenizer;
use crate::sampling::{SampleParams, sample_lines, sample_lines_cancel};
use crate::tokenizer::{BBPETokenizer, BpeState, LoadOptions, SamplingMeta, SaveOptions};
use crate::trainer::BBPETrainer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use crate::reader::{CorpusReader, CorpusStats};
// 引入

use crate::trainer::{RuntimeConfig, TrainerConfig};

use crate::tokenizer::Encoding as RustEncoding;

// 重命名 Python 暴露的结构体，实现与底层解耦
#[pyclass(name = "Encoding", module = "bbpe_tokenizer.bbpe_rs", get_all)]
pub struct PyEncoding {
    pub ids: Vec<i32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
}

// 优雅的类型转换 trait
impl From<RustEncoding> for PyEncoding {
    fn from(enc: RustEncoding) -> Self {
        Self {
            ids: enc.ids,
            tokens: enc.tokens,
            offsets: enc.offsets,
        }
    }
}

// 【统一的 Flavor 枚举】
#[pyclass(
    name = "PatternFlavor",
    module = "bbpe_tokenizer.bbpe_rs",
    from_py_object
)]
#[derive(Clone)]
pub enum PyPatternFlavor {
    Gpt4(),
    Llama3(),
    Custom { pattern: String },
}

#[pymethods]
impl PyPatternFlavor {
    #[staticmethod]
    pub fn gpt4() -> Self {
        Self::Gpt4()
    }
    #[staticmethod]
    pub fn llama3() -> Self {
        Self::Llama3()
    }
    #[staticmethod]
    pub fn custom(pattern: String) -> Self {
        Self::Custom { pattern }
    }
}

// =========================================================
// 暴露 PreTokenizer 给 Python
// =========================================================
/// 预分词器（pattern 的直接载体）。
///
/// 存在意义：此前 Python 侧**只能训练时设 pattern、却无法在编码前验证它**。
/// pattern 的"全文覆盖"约束（未被任何匹配覆盖的字节会被静默丢弃，见
/// `PreTokenizer::tokenize_with_checked`）只能在 Rust 侧触发，用户拿到的是
/// "某些字符凭空消失"的编码结果。本类提供 `pre_tokenize` /
/// `pre_tokenize_checked` 两个预览入口，把这件事变成可观测、可断言的。
#[pyclass(name = "PreTokenizer", module = "bbpe_tokenizer.bbpe_rs")]
pub struct PyPreTokenizer {
    pub(crate) inner: PreTokenizer,
}

#[pymethods]
impl PyPreTokenizer {
    #[new]
    #[pyo3(signature = (pattern, max_chunk_bytes=None))]
    fn new(pattern: &str, max_chunk_bytes: Option<usize>) -> PyResult<Self> {
        let inner = match max_chunk_bytes {
            Some(n) => PreTokenizer::with_max_chunk_bytes(pattern, n),
            None => PreTokenizer::new(pattern),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// 预览切分结果（**非严格**，与 Rust `PreTokenizer::pre_tokenize` 同口径）：
    /// 未被 pattern 覆盖的间隙会被**静默丢弃**。要求"必须全覆盖"请用
    /// [`Self::pre_tokenize_checked`]。
    ///
    /// 返回 `[(bytes, start_offset), ...]`；`bytes` 是该片的原始字节（UTF-8，
    /// 切分点保证落在字符边界上）。
    fn pre_tokenize<'py>(&self, py: Python<'py>, text: &str) -> Vec<(Bound<'py, PyBytes>, usize)> {
        self.inner
            .pre_tokenize(text)
            .into_iter()
            .map(|(bytes, at)| (PyBytes::new(py, bytes), at))
            .collect()
    }

    /// 严格切分：pattern 未覆盖全部文本时抛 `RuntimeError`，消息里带缺失的
    /// 字节区间与内容样本。
    ///
    /// 这是 Python 侧**唯一**能事前验证自定义 pattern 的入口 —— 训练与编码
    /// 的 `*_checked` 路径要到真正处理语料时才会暴露问题。
    fn pre_tokenize_checked<'py>(
        &self,
        py: Python<'py>,
        text: &str,
    ) -> PyResult<Vec<(Bound<'py, PyBytes>, usize)>> {
        let mut out = Vec::new();
        self.inner
            .tokenize_with_checked(text, |bytes, at| {
                out.push((PyBytes::new(py, bytes), at));
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }

    #[getter]
    fn pattern(&self) -> &str {
        self.inner.pattern()
    }

    #[getter]
    fn max_chunk_bytes(&self) -> usize {
        self.inner.max_chunk_bytes()
    }
}

// =========================================================
// 暴露 TrainerConfig 给 Python
// =========================================================
#[pyclass(
    name = "TrainerConfig",
    module = "bbpe_tokenizer.bbpe_rs",
    from_py_object
)]
#[derive(Clone)]
pub struct PyTrainerConfig {
    pub(crate) inner: TrainerConfig,
}

#[pymethods]
impl PyTrainerConfig {
    #[new]
    pub fn new(vocab_size: usize) -> Self {
        Self {
            inner: TrainerConfig::new(vocab_size),
        }
    }

    pub fn with_pattern(
        mut slf: PyRefMut<'_, Self>,
        flavor: PyPatternFlavor,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let pattern_str = match &flavor {
            PyPatternFlavor::Gpt4() => crate::pre_tokenizer::PATTERN_GPT4.to_string(),
            PyPatternFlavor::Llama3() => crate::pre_tokenizer::PATTERN_LLAMA3.to_string(),
            PyPatternFlavor::Custom { pattern } => pattern.clone(),
        };
        slf.inner.pre_tokenizer_pattern = pattern_str;
        Ok(slf)
    }

    pub fn with_min_frequency(
        mut slf: PyRefMut<'_, Self>,
        freq: u32,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.min_frequency = freq;
        Ok(slf) // 使得可以在 Python 端进行连缀调用：config.with_min_frequency(2)
    }

    pub fn with_special_tokens(
        mut slf: PyRefMut<'_, Self>,
        tokens: Vec<String>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.special_tokens = tokens;
        Ok(slf)
    }

    // ---------- 读取面 ----------
    // 此前 Python 侧**只写不读**：设了 pattern/min_frequency 之后无法取回，
    // 断言/日志/复现都只能靠 Python 变量自己另存一份，很容易与实际生效值脱节。
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size
    }

    #[getter]
    fn min_frequency(&self) -> u32 {
        self.inner.min_frequency
    }

    #[getter]
    fn special_tokens(&self) -> Vec<String> {
        self.inner.special_tokens.clone()
    }

    /// 当前生效的预分词正则原文（`with_pattern` 写入的那份）。
    #[getter]
    fn pattern(&self) -> &str {
        &self.inner.pre_tokenizer_pattern
    }
}

// =========================================================
// 暴露 RuntimeConfig 给 Python
// =========================================================
#[pyclass(
    name = "RuntimeConfig",
    module = "bbpe_tokenizer.bbpe_rs",
    from_py_object
)]
// 同时派生 Default：clippy::new_without_default 指出存在 `new()` 却无 `Default`。
// 内层 `RuntimeConfig` 已实现 Default，故派生即可。
#[derive(Clone, Default)]
pub struct PyRuntimeConfig {
    pub(crate) inner: RuntimeConfig,
}

#[pymethods]
impl PyRuntimeConfig {
    /// 构造默认配置。
    ///
    /// 同时实现 [`Default`]（clippy::new_without_default）：pyo3 侧不需要它，
    /// 但"有无默认构造"是类型是否自洽的一部分，补上更符合 Rust 惯例，
    /// 也让 Python 之外的 Rust 调用方不必重复默认值。
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_batch_size(
        mut slf: PyRefMut<'_, Self>,
        size: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.batch_size = size;
        Ok(slf)
    }

    pub fn with_num_shards(
        mut slf: PyRefMut<'_, Self>,
        shards: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.num_shards = shards;
        Ok(slf)
    }

    pub fn with_parallel_index(
        mut slf: PyRefMut<'_, Self>,
        parallel: bool,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.parallel_index_build = parallel;
        Ok(slf)
    }

    /// 单 chunk 字节上限（`0` = 不切分）。
    ///
    /// m4 之后该值**真正参与预分词切分**（`trainer.rs` 把它接到
    /// `PreTokenizer::with_max_chunk_bytes`，并写进 `meta.train.max_chunk_bytes`），
    /// 但 Python 侧此前完全没有入口 —— 于是同一个模型用 CLI 训练与用 Python
    /// 训练会得到不同的分块行为，且无法在 Python 里复现前者。本方法补齐该口子。
    pub fn with_max_chunk_bytes(
        mut slf: PyRefMut<'_, Self>,
        bytes: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.max_chunk_bytes = bytes;
        Ok(slf)
    }

    // ---------- 读取面 ----------
    #[getter]
    fn batch_size(&self) -> usize {
        self.inner.batch_size
    }

    #[getter]
    fn num_shards(&self) -> usize {
        self.inner.num_shards
    }

    #[getter]
    fn parallel_index(&self) -> bool {
        self.inner.parallel_index_build
    }

    #[getter]
    fn max_chunk_bytes(&self) -> usize {
        self.inner.max_chunk_bytes
    }
}

// ---------- 新增 Python 侧的 Reader 类 ----------
/// 行迭代器：Python 侧 `for line in reader.iter_lines()` 的载体。
///
/// `unsendable`：底层是 `Box<dyn Iterator<Item = String>>`（无 `Send` 约束），
/// 强制要求线程亲和，避免把文件句柄搬过线程后读到未定义状态。
#[pyclass(name = "LineIterator", module = "bbpe_tokenizer.bbpe_rs", unsendable)]
pub struct PyLineIter {
    inner: Box<dyn Iterator<Item = String>>,
}

#[pymethods]
impl PyLineIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<String> {
        slf.inner.next()
    }
}

/// 两个 Reader 类共用的行迭代构造（错误映射只此一处）。
fn iter_lines_of(reader: &CorpusReader) -> PyResult<PyLineIter> {
    reader
        .iter_lines()
        .map(|inner| PyLineIter { inner })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyclass(name = "TextFileReader", module = "bbpe_tokenizer.bbpe_rs")]
pub struct PyTextFileReader {
    pub(crate) inner: CorpusReader,
}
#[pymethods]
impl PyTextFileReader {
    #[new]
    pub fn new(filepath: &str) -> Self {
        Self {
            inner: CorpusReader::Text {
                filepath: filepath.to_string(),
            },
        }
    }

    /// 逐行读取原始语料（只读预览；训练仍走 `BBPETrainer.train`，
    /// 那条路径带批量聚合与统计，语义不同）。
    fn iter_lines(&self) -> PyResult<PyLineIter> {
        iter_lines_of(&self.inner)
    }

    #[getter]
    fn filepath(&self) -> String {
        self.inner.filepath().to_string()
    }

    /// `"text"` —— 与 Rust `CorpusReader::kind()` 同源。
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    #[getter]
    fn file_size(&self) -> u64 {
        self.inner.file_size()
    }
}

#[pyclass(name = "JsonlFileReader", module = "bbpe_tokenizer.bbpe_rs")]
pub struct PyJsonlFileReader {
    pub(crate) inner: CorpusReader,
}

#[pymethods]
impl PyJsonlFileReader {
    #[staticmethod]
    pub fn static_text(filepath: &str) -> Self {
        Self {
            inner: CorpusReader::JsonlStaticText {
                filepath: filepath.to_string(),
            },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (filepath, separator="\n"))]
    pub fn static_conversations(filepath: &str, separator: &str) -> Self {
        Self {
            inner: CorpusReader::JsonlStaticConversations {
                filepath: filepath.to_string(),
                sep: separator.to_string(),
            },
        }
    }

    /// 动态模式引擎：完美消化用户的 Schema
    ///
    /// n1：**按 dict 的插入顺序**构建规则。
    /// 原签名为 `HashMap<String, (String, String)>`，PyO3 把 Python dict 转成
    /// Rust HashMap 时会丢弃插入顺序；而 `CorpusReader::JsonlDynamic` 的
    /// schema_rules 语义是"按顺序逐个尝试、命中即提取"⟹ 多字段时到底命中哪个
    /// 字段会随 Rust HashMap 的随机迭代序漂移，同一份脚本两次运行可能读出
    /// 不同结果——这与项目"确定性/可复现"的底线直接冲突。
    /// 改为接收 `PyDict` 并按插入序遍历，顺序即用户书写顺序。
    #[staticmethod]
    pub fn dynamic(filepath: &str, schema: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut schema_rules = Vec::new();
        for (_key, value) in schema.iter() {
            let (path, ty): (String, String) = value.extract()?;
            let pointers: Vec<String> = path
                .trim_matches('/')
                .split('/')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
            schema_rules.push((pointers, ty.to_lowercase()));
        }

        Ok(Self {
            inner: CorpusReader::JsonlDynamic {
                filepath: filepath.to_string(),
                schema_rules,
            },
        })
    }

    /// 逐行读取（按该 reader 的提取规则返回 `str`：static_text 取 `text` 字段、
    /// static_conversations 拼接多轮、dynamic 按 schema 命中即取）。
    fn iter_lines(&self) -> PyResult<PyLineIter> {
        iter_lines_of(&self.inner)
    }

    #[getter]
    fn filepath(&self) -> String {
        self.inner.filepath().to_string()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    #[getter]
    fn file_size(&self) -> u64 {
        self.inner.file_size()
    }
}

// 自动类型转换枚举：用于接受 Python 任意类型的 Reader
#[derive(FromPyObject)]
pub enum PyReaderInput<'py> {
    Text(PyRef<'py, PyTextFileReader>),
    Jsonl(PyRef<'py, PyJsonlFileReader>),
}

/// R6-1 P2：可从任意 Python 线程 / 信号处理器调用的取消令牌。
///
/// 与训练器解耦的好处：信号处理器里调 `token.cancel()` 不会撞 PyO3 的
/// 对象借用（训练态已在 worker 线程，Python 侧只共享这个 Arc）。
#[pyclass(name = "CancellationToken", module = "bbpe_tokenizer.bbpe_rs")]
pub struct PyCancelToken {
    inner: Arc<CancelToken>,
}

#[pymethods]
impl PyCancelToken {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(CancelToken::new()),
        }
    }

    /// 请求取消。可重复调用；取消原因以第一次为准。
    fn cancel(&self) {
        self.inner.cancel(CancelReason::User);
    }

    #[getter]
    fn cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    #[getter]
    fn reason(&self) -> &'static str {
        reason_str(self.inner.reason())
    }

    #[getter]
    fn merges_done(&self) -> usize {
        self.inner.progress()
    }
}

/// `py.check_signals()` 只在主线程生效。非主线程调用 `train()` 时发一条
/// 自解释的告警，把「Ctrl+C 没反应」变成「文档告诉我该传 token」。
fn is_main_thread(py: Python<'_>) -> bool {
    let Ok(threading) = py.import("threading") else {
        return true;
    };
    match (
        threading.call_method0("current_thread"),
        threading.call_method0("main_thread"),
    ) {
        (Ok(cur), Ok(main)) => cur.is(&main),
        _ => true,
    }
}

// ---------------- R6-1 P1：worker 线程 + 主线程信号泵 ----------------

/// 与目标文件同目录的临时落盘路径。
///
/// 保证 `rename` 在同一文件系统内原子完成：采样先写 `.part` 暂存文件，
/// 完整成功后才提交到目标路径。取消 / 采样失败 / 预检失败时目标文件
/// **完全不被触碰**，用户已有的同名文件不会被截断成半截产物。
fn staging_path(target: &std::path::Path) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let name = target
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());
    let dir = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    dir.join(format!(".{name}.part.{}.{nanos}", std::process::id()))
}

/// 采样/预检/训练**整体**在 worker 内完成，主线程信号泵覆盖全流程。
/// 临时样本文件的 `TempFileGuard` 是 worker 的局部变量，worker 结束时
/// （成功 / 失败 / 取消）自动 drop —— 不依赖 `wait_cleanup()` 回压。
type TrainOutcome = (
    Result<(), BbpeError>,
    BBPETrainer,
    Arc<CorpusStats>,
    Option<SamplingMeta>, // 只在训练成功时据实写入 G1 出身信息
    bool,                 // precheck_skipped：可用内存读不到，需主线程告警
);

/// 训练态整体 move 进 worker 线程；`inner` 为 `None` 表示「训练进行中」或
/// 「训练已取消、状态已丢弃」。两种情况都不存在可保存的半截 state。
///
/// 取消后不 join worker（让几十 GB 训练态在后台释放），句柄暂存在 `cleanup`，
/// 由 `wait_cleanup()` 或下一次 `train()`/`reset()` 回填压，避免旧状态与
/// 新一轮训练同时占据内存导致双份峰值 OOM。
#[pyclass(name = "BBPETrainer", module = "bbpe_tokenizer.bbpe_rs")]
pub struct PyBBPETrainer {
    inner: Mutex<Option<BBPETrainer>>,
    cleanup: Mutex<Option<JoinHandle<TrainOutcome>>>,
    trainer_config: TrainerConfig,
    runtime_config: RuntimeConfig,
}

impl PyBBPETrainer {
    fn lock_inner(&self) -> MutexGuard<'_, Option<BBPETrainer>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_cleanup(&self) -> MutexGuard<'_, Option<JoinHandle<TrainOutcome>>> {
        self.cleanup.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 等待上一次取消的后台释放完成；没有未完成任务时是 no-op。
    fn join_cleanup(&self) {
        if let Some(handle) = self.lock_cleanup().take() {
            let _ = handle.join();
        }
    }

    fn no_state_error() -> PyErr {
        PyRuntimeError::new_err(
            "没有可用的训练状态：训练进行中，或上次训练已被取消。\
             请调用 reset() 或新建 trainer",
        )
    }
}

fn reason_str(r: CancelReason) -> &'static str {
    match r {
        CancelReason::Signal => "signal",
        CancelReason::User => "user",
        CancelReason::Timeout => "timeout",
        CancelReason::None => "unknown",
    }
}

/// 构造取消异常：优先**原样**返回 signal handler 抛出的 PyErr，再附加
/// provenance 属性。不落盘、不改模型契约。
fn build_cancel_error(
    py: Python<'_>,
    signal_err: Option<PyErr>,
    token: &CancelToken,
    stats: &CorpusStats,
    started: Instant,
) -> PyErr {
    let err = signal_err.unwrap_or_else(|| match token.reason() {
        CancelReason::Signal => pyo3::exceptions::PyKeyboardInterrupt::new_err("训练已取消"),
        CancelReason::Timeout => {
            TrainingCancelled::new_err("训练超时已取消；请调用 reset() 或新建 trainer 重新训练")
        }
        _ => TrainingCancelled::new_err("训练已取消；请调用 reset() 或新建 trainer 重新训练"),
    });

    let (total_lines, total_bytes, skipped, replaced) = stats.snapshot();
    let d = PyDict::new(py);
    let _ = d.set_item("total_lines", total_lines);
    let _ = d.set_item("total_bytes", total_bytes);
    let _ = d.set_item("skipped_lines", skipped);
    let _ = d.set_item("replacement_chars", replaced);

    let value = err.value(py);
    let _ = value.setattr("merges_done", token.progress());
    let _ = value.setattr("reason", reason_str(token.reason()));
    let _ = value.setattr("elapsed_secs", started.elapsed().as_secs_f64());
    let _ = value.setattr("corpus_stats", &d);
    err
}

/// 推理侧可中断阈值：短任务（P99 微秒级）走零开销原路径，只有长文本/大批
/// decode 才值得起 worker 线程。
const INFERENCE_INTERRUPT_BYTES: usize = 1 << 20;
const INFERENCE_INTERRUPT_IDS: usize = 200_000;

/// 长推理任务的统一可中断执行器。
///
/// - 短输入：`py.detach` 原路径，行为与 R4-16 完全一致。
/// - 长输入：worker 线程执行纯 Rust 计算，主线程 20ms 信号泵释放 GIL；
///   收到 SIGINT 时**立即返回原始异常**，worker 结果被丢弃（`JoinHandle`
///   drop 即 detach）。
///
/// encode/decode 是纯函数、不修改任何共享状态、无 I/O，因此「取消 = 放弃
/// 本次结果」是安全的；worker 不持有任何 Python 对象，进程退出不受影响。
/// 这与 HF tokenizers `StreamingIter::drop → batch.cancel()` 的取舍一致。
fn run_inference_interruptible<R, F>(py: Python<'_>, long_enough: bool, work: F) -> PyResult<R>
where
    R: Send + 'static,
    F: FnOnce() -> PyResult<R> + Send + 'static,
{
    if !long_enough {
        return py.detach(work);
    }

    let handle = std::thread::spawn(work);
    loop {
        py.detach(|| std::thread::sleep(Duration::from_millis(20)));
        if let Some(Err(e)) = Python::try_attach(|py| py.check_signals()) {
            return Err(e);
        }
        if handle.is_finished() {
            break;
        }
    }
    handle
        .join()
        .map_err(|_| PyRuntimeError::new_err("推理线程发生 panic"))?
}

#[pymethods]
impl PyBBPETrainer {
    #[new]
    #[pyo3(signature = (trainer_config, runtime_config=None))]
    pub fn new(
        trainer_config: &PyTrainerConfig,
        runtime_config: Option<&PyRuntimeConfig>,
    ) -> PyResult<Self> {
        let tc = trainer_config.inner.clone();
        let rt = runtime_config.map(|c| c.inner.clone()).unwrap_or_default();

        // pre_tokenizer_pattern 已由 TrainerConfig 携带，BBPETrainer::new 内部会读取。
        let inner = BBPETrainer::new(tc.clone(), Some(rt.clone()))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Mutex::new(Some(inner)),
            cleanup: Mutex::new(None),
            trainer_config: tc,
            runtime_config: rt,
        })
    }

    /// 训练（R6-1：释放 GIL + 全流程可中断 + 取消即丢弃训练态）。
    ///
    /// **采样预读 → OOM 预检 → 训练**整体在 worker 线程执行；主线程从
    /// spawn 起就进入 pump loop，每 20ms 用 `py.detach` 放开 GIL 睡一拍，
    /// 再 `check_signals()` 让 Python 信号处理器有机会运行。因此：
    ///   ① 其它 Python 线程能推进（m10 已闭环）；
    ///   ② Ctrl+C / token / timeout 在**采样预读阶段**同样生效，
    ///      `sample_lines_cancel` 每 4096 行检查令牌，底层真的停下；
    ///   ③ 取消不 join worker，异常立即返回，大训练态在后台释放。
    ///
    /// P3：`sample_rate` / `sample_seed` / `max_bytes` / `heldout_rate` /
    /// `heldout_seed` / `heldout_out` / `sample_out` 复用 CLI 的
    /// `sampling::sample_lines` 语义；`oom_precheck`（默认开启）按
    /// `9.2 × 训练语料字节数` 拦截必 OOM 的配置。可用内存读不到时显式
    /// 告警而非静默跳过。**预检失败绝不删除任何用户输入文件**（R4-1）。
    // PyO3 的 Python kwargs 必须扁平展开；拆成 Rust 结构体只会让 Python API
    // 变别扭。参数多来自「取消 + 采样 + 预检」三组各自独立的可选能力。
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (reader, cancel_token=None, timeout=None, sample_rate=None,
                        sample_seed=42, max_bytes=None, oom_precheck=None,
                        heldout_rate=0.0, heldout_seed=43, heldout_out=None,
                        sample_out=None))]
    pub fn train(
        &self,
        py: Python<'_>,
        reader: PyReaderInput,
        cancel_token: Option<&PyCancelToken>,
        timeout: Option<f64>,
        sample_rate: Option<f64>,
        sample_seed: u64,
        max_bytes: Option<u64>,
        oom_precheck: Option<bool>,
        heldout_rate: f64,
        heldout_seed: u64,
        heldout_out: Option<String>,
        sample_out: Option<String>,
    ) -> PyResult<Py<PyDict>> {
        if let Some(t) = timeout {
            if !t.is_finite() || t <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "timeout 必须是 > 0 的有限秒数",
                ));
            }
        }
        if let Some(rate) = sample_rate {
            if !rate.is_finite() || !(0.0 < rate && rate <= 1.0) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "sample_rate 必须落在 (0, 1]",
                ));
            }
        }
        if !heldout_rate.is_finite() || !(0.0..1.0).contains(&heldout_rate) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "heldout_rate 必须落在 [0, 1)",
            ));
        }
        if heldout_rate > 0.0 {
            if sample_rate.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "heldout_rate > 0 时必须同时指定 sample_rate",
                ));
            }
            if heldout_out.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "heldout_rate > 0 时必须提供 heldout_out（否则命中行会被静默丢弃）",
                ));
            }
        }
        if !is_main_thread(py) {
            warn_python(
                "BBPETrainer.train() 在非主线程调用：Ctrl+C 不会自动中断训练。\
                 请传入 CancellationToken 或设置 timeout=，\
                 或在主线程用 `with bbpe_tokenizer.interruptible() as tok` 包住后台训练。",
            );
        }

        // 回压：等上一次取消的后台释放结束，避免旧/新训练态同时占内存。
        self.join_cleanup();

        let reader_owned = match &reader {
            PyReaderInput::Text(r) => r.inner.clone(),
            PyReaderInput::Jsonl(r) => r.inner.clone(),
        };

        // 路径别名防护：采样/held-out 输出绝不能覆盖原始语料（CLI 同款）。
        let src_path = reader_owned.filepath();
        for (flag, p) in [("sample_out", &sample_out), ("heldout_out", &heldout_out)] {
            if let Some(p) = p {
                if same_path(src_path, p) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{flag} 不能指向原始输入语料"
                    )));
                }
            }
        }
        if let (Some(a), Some(b)) = (&sample_out, &heldout_out) {
            if same_path(a, b) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "sample_out 与 heldout_out 不能是同一个文件",
                ));
            }
        }

        let mut trainer = self.lock_inner().take().ok_or_else(Self::no_state_error)?;

        // 输入契约必须覆盖 Python 入口（规范 §2.4 / §6.6）。主线程保留一份
        // Arc，取消时无需 join worker 就能读到语料统计。
        let stats = Arc::new(CorpusStats::default());
        let stats_main = stats.clone();
        // 外部令牌与 worker 共享同一个 Arc；未传则内部新建。
        // 多个取消源（信号 / timeout / 外部 cancel）以先到者为准。
        let token = cancel_token
            .map(|t| t.inner.clone())
            .unwrap_or_else(|| Arc::new(CancelToken::new()));
        let worker_token = token.clone();
        // 取消时主线程不 join worker；若进程恰好立即退出，worker 可能来不及
        // drop 守卫。共享临时路径让主线程兜底删除（remove_file 幂等）。
        let temp_sample_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let temp_sample_path_worker = temp_sample_path.clone();

        let handle: JoinHandle<TrainOutcome> = std::thread::spawn(move || {
            // 采样/预检/训练都在 worker 内；主线程 pump 从 spawn 起覆盖全程，
            // 所以采样预读阶段同样能被 Ctrl+C / token / timeout 中断。
            let mut sample_guard: Option<TempFileGuard> = None;
            let mut sampling_meta: Option<SamplingMeta> = None;
            let mut active_reader = reader_owned.clone();
            let mut precheck_skipped = false;

            let prep: Result<(), BbpeError> = (|| {
                if let Some(rate) = sample_rate {
                    let src = reader_owned.filepath().to_string();

                    // 落盘策略：**先写同目录 .part 暂存文件，采样完整成功后才
                    // rename 提交**。取消 / IO 错误 / 预检失败都不会把用户已有的
                    // sample_out / heldout_out 截断成半截产物。
                    let train_target: Option<PathBuf> = sample_out.as_deref().map(PathBuf::from);
                    let train_staging = match &train_target {
                        Some(t) => staging_path(t),
                        None => temp_path("bbpe_py_sample", ".jsonl"),
                    };
                    let mut train_guard = Some(TempFileGuard::new(train_staging.clone()));
                    if train_target.is_none() {
                        *temp_sample_path_worker
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) = Some(train_staging.clone());
                    }
                    // held-out 只在真正启用时才落盘，避免 rate=0 时留下空文件。
                    let held_target: Option<PathBuf> = if heldout_rate > 0.0 {
                        heldout_out.as_deref().map(PathBuf::from)
                    } else {
                        None
                    };
                    let held_staging = held_target.as_ref().map(|t| staging_path(t));
                    let mut held_guard = held_staging.clone().map(TempFileGuard::new);

                    let mut input = BufReader::with_capacity(1 << 20, File::open(&src)?);
                    let mut out = BufWriter::with_capacity(1 << 20, File::create(&train_staging)?);
                    let mut held_w = match &held_staging {
                        Some(p) => Some(BufWriter::with_capacity(1 << 20, File::create(p)?)),
                        None => None,
                    };
                    let s = sample_lines_cancel(
                        &mut input,
                        &mut out,
                        held_w.as_mut(),
                        SampleParams {
                            rate,
                            seed: sample_seed,
                            heldout_rate,
                            heldout_seed,
                            max_bytes: max_bytes.unwrap_or(u64::MAX),
                        },
                        Some(worker_token.as_ref()),
                    )
                    .map_err(|e| {
                        if e.kind() == std::io::ErrorKind::Interrupted {
                            // 采样阶段取消：与训练取消统一走 Cancelled 分支
                            BbpeError::Cancelled {
                                merges_done: 0,
                                reason: worker_token.reason(),
                            }
                        } else {
                            BbpeError::IoError(e)
                        }
                    })?;
                    out.flush()?;
                    drop(out);
                    if let Some(w) = held_w.as_mut() {
                        w.flush()?;
                    }
                    drop(held_w);

                    // 采样完整成功：原子提交到用户指定路径
                    let committed_train: PathBuf = match train_target.as_ref() {
                        Some(target) => {
                            let g = train_guard.take().expect("staging guard 必然存在");
                            let p = g.keep();
                            if let Err(e) = std::fs::rename(&p, target) {
                                let _ = std::fs::remove_file(&p);
                                return Err(BbpeError::IoError(e));
                            }
                            target.clone()
                        }
                        None => {
                            // 自动临时文件：守卫保留到 worker 结束再删除
                            sample_guard = train_guard.take();
                            train_staging.clone()
                        }
                    };
                    if let Some(target) = held_target.as_ref() {
                        let g = held_guard.take().expect("held-out staging guard 必然存在");
                        let p = g.keep();
                        if let Err(e) = std::fs::rename(&p, target) {
                            let _ = std::fs::remove_file(&p);
                            return Err(BbpeError::IoError(e));
                        }
                    }

                    active_reader =
                        reader_owned.with_filepath(committed_train.to_string_lossy().into_owned());
                    sampling_meta = Some(SamplingMeta {
                        rate,
                        seed: sample_seed,
                        heldout_rate,
                        heldout_seed,
                    });
                    tracing::info!(
                        "Python 采样: 总 {} 行 ⟹ 训练 {} 行 / held-out {} 行（{}，seed={}）",
                        s.total_lines,
                        s.train_lines,
                        s.held_lines,
                        human_bytes(s.train_bytes),
                        sample_seed
                    );
                }

                if oom_precheck.unwrap_or(true) {
                    let bytes = active_reader.file_size();
                    let avail = available_ram_bytes();
                    if avail.is_none() {
                        precheck_skipped = true;
                    }
                    if exceeds_memory_budget(bytes, avail) {
                        let a = avail.unwrap_or(0);
                        let suggest = ((a as f64 * 0.8 / PEAK_MEM_FACTOR)
                            / reader_owned.file_size().max(1) as f64)
                            .clamp(0.01, 1.0);
                        return Err(BbpeError::MemoryBudgetExceeded(format!(
                            "语料装不进内存：训练语料 {} × {PEAK_MEM_FACTOR} ≈ {}，可用内存约 {}。\n\
                             原始输入文件**未做任何改动**；建议传 sample_rate={:.3} 重新训练，\
                             或显式 oom_precheck=False 自行承担 OOM 风险。",
                            human_bytes(bytes),
                            human_bytes((bytes as f64 * PEAK_MEM_FACTOR) as u64),
                            human_bytes(a),
                            suggest,
                        )));
                    }
                }
                Ok(())
            })();

            let result = match prep {
                Err(e) => Err(e),
                Ok(()) => trainer.train_with_cancel(
                    &active_reader,
                    Some(stats.clone()),
                    Some(worker_token.as_ref()),
                ),
            };
            // 守卫是 worker 局部变量：worker 结束（含取消路径）即清理临时
            // 样本，不需要主线程 join，也不依赖 wait_cleanup()。
            drop(sample_guard);
            (result, trainer, stats, sampling_meta, precheck_skipped)
        });

        let started = Instant::now();
        let deadline = timeout.map(|secs| started + Duration::from_secs_f64(secs));
        let mut signal_err: Option<PyErr> = None;
        loop {
            py.detach(|| std::thread::sleep(Duration::from_millis(20)));
            // try_attach 兼顾解释器正在 finalization 的情形：attach 不上就
            // 兜底取消，避免主线程卡在已无法处理信号的解释器里。
            match Python::try_attach(|py| py.check_signals()) {
                Some(Ok(())) => {}
                Some(Err(e)) => {
                    if signal_err.is_none() {
                        signal_err = Some(e);
                    }
                    token.cancel(CancelReason::Signal);
                }
                None => token.cancel(CancelReason::Signal),
            }
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    token.cancel(CancelReason::Timeout);
                }
            }
            if handle.is_finished() || token.is_cancelled() {
                break;
            }
        }

        if !handle.is_finished() {
            // 取消路径：detach worker，主线程立即返回；worker 继续按自己的
            // 栈顺序 drop 训练态。句柄留给 wait_cleanup() 回填压。
            *self.lock_cleanup() = Some(handle);
            // 兜底清理临时样本：进程可能在 worker drop 守卫之前就退出。
            if let Some(p) = temp_sample_path
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
            {
                let _ = std::fs::remove_file(p);
            }
            return Err(build_cancel_error(
                py,
                signal_err,
                &token,
                &stats_main,
                started,
            ));
        }

        let (result, trained, stats_back, sampling_meta, precheck_skipped) = handle
            .join()
            .map_err(|_| PyRuntimeError::new_err("训练线程发生 panic，训练已中止"))?;

        if precheck_skipped {
            warn_python(
                "OOM 预检已跳过：无法读取可用内存（MemAvailable / MemTotal / cgroup 均不可用）；\
                 请自行评估内存风险",
            );
        }

        match result {
            Ok(()) => {
                let mut trained = trained;
                // Python 绑定属核心（v2 剖面，规范 §9.1），标记 profile（G1）
                trained.state.meta_mut().profile = Some("v2".to_string());
                // P3：采样出身信息据实写入，不能让采样产物伪装成全量训练
                if let Some(s) = sampling_meta {
                    trained.state.meta_mut().sampling = Some(s);
                }
                *self.lock_inner() = Some(trained);

                // 静默行为必须可见（G4）。m8：统一走 warn_python
                // （warnings.warn → stderr），不再依赖 builtins.print。
                let (_total_lines, _total_bytes, skipped, replaced) = stats_main.snapshot();
                if skipped > 0 || replaced > 0 {
                    let msg = format!(
                        "bbpe_tokenizer.BBPETrainer.train: skipped_lines={skipped}, \
                         replacement_chars={replaced}（已写入模型 meta.corpus；读取行为不变，\
                         语义见规范 §6.2/§6.3）"
                    );
                    warn_python(&msg);
                }
                // 训练统计回传给调用方（此前 worker 交回的 stats 被直接丢弃）：
                // skipped_lines / io_errors 是"语料被改动/截断"的唯一信号，
                // 只发 warning 不够 —— 机器调用方需要能断言。
                build_stats_dict(py, &stats_back)
            }
            Err(e) => {
                if matches!(e, BbpeError::Cancelled { .. }) {
                    // 取消即丢弃：状态槽保持 None，物理上无半截模型可保存。
                    return Err(build_cancel_error(
                        py,
                        signal_err,
                        &token,
                        &stats_main,
                        started,
                    ));
                }
                // 普通错误沿用 P0 行为：状态仍在，A1 守卫会挡二次训练。
                *self.lock_inner() = Some(trained);
                Err(PyRuntimeError::new_err(e.to_string()))
            }
        }
    }

    #[pyo3(signature = (filepath, no_meta=false))]
    pub fn save(&self, filepath: &str, no_meta: bool) -> PyResult<()> {
        let guard = self.lock_inner();
        let trainer = guard.as_ref().ok_or_else(Self::no_state_error)?;
        trainer
            .save_with_options(filepath, SaveOptions { no_meta })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// 替换实际使用的预分词器（R4-23：同步 state / trainer_config /
    /// runtime_config 三处记录，避免"契约记录 ≠ 实际切分行为"）。
    ///
    /// 典型用法：`RuntimeConfig` 只暴露 `max_chunk_bytes`，而想连 pattern 一起
    /// 换、或在训练前先验证 pattern，用本方法 + `PreTokenizer`。
    ///
    /// 注意：对**已训练**的实例调用只影响后续的 `reset()` 重建，不会改写已产出
    /// 的 merges；要按新 pattern 重训请先 `reset()`。取消后状态已丢弃时返回
    /// `RuntimeError`（没有可修改的实例）。
    pub fn set_pre_tokenizer(&self, pre_tokenizer: &PyPreTokenizer) -> PyResult<()> {
        let mut guard = self.lock_inner();
        let trainer = guard.as_mut().ok_or_else(Self::no_state_error)?;
        trainer.set_pre_tokenizer(pre_tokenizer.inner.clone());
        Ok(())
    }

    /// 等待上一次取消的后台内存释放完成。从未取消时是 no-op；
    /// `train()` / `reset()` 会隐式调用它做回压，也可显式调用。
    pub fn wait_cleanup(&self) {
        self.join_cleanup();
    }

    /// 回到"未训练"状态，使其可再次训练（见 [`train`] 的幂等性说明）。
    ///
    /// B4：`BbpeError::AlreadyTrained` 的处置文案写的是"新建一个 trainer；或
    /// 显式调用 `reset()`"，而 Python 恰恰是 A1 最自然的触发路径（Notebook 里
    /// 换个语料重训）。取消后训练态已丢弃，本方法依构造时的 config 重建。
    pub fn reset(&self) {
        self.join_cleanup();
        *self.lock_inner() = None;
        if let Ok(fresh) = BBPETrainer::new(
            self.trainer_config.clone(),
            Some(self.runtime_config.clone()),
        ) {
            *self.lock_inner() = Some(fresh);
        }
    }

    /// 当前是否已训练过（`merges` 非空）。供调用方在复用 trainer 前自查，
    /// 避免先撞上 [`BbpeError::AlreadyTrained`] 再想办法。
    pub fn is_trained(&self) -> bool {
        self.lock_inner().as_ref().is_some_and(|t| t.is_trained())
    }
}

#[pyclass(name = "BBPETokenizer", module = "bbpe_tokenizer.bbpe_rs")]
pub struct PyBBPETokenizer {
    // R6-1：长编码/解码要交给 worker 线程，Arc 让「共享模型」零拷贝；
    // 其余方法经 Deref 使用，行为不变。
    inner: Arc<BBPETokenizer>,
}

#[pymethods]
impl PyBBPETokenizer {
    #[new]
    #[pyo3(signature = (filepath, allow_legacy=false, validate=true))]
    pub fn new(filepath: &str, allow_legacy: bool, validate: bool) -> PyResult<Self> {
        // 迁移三件套之 Python 侧（规范 §7.5）：allow_legacy=true 时可加载 v0 旧模型
        // （缺 pre_tokenizer_pattern 时回退 DEFAULT_PATTERN 并打印警告）
        //
        // `validate` 对齐 Rust `LoadOptions`：默认 true（V1–V7 校验），仅诊断
        // 破损模型时才需要关掉 —— 关掉后 `cut`/`save` 仍会拒绝自相矛盾的 state。
        let inner = crate::tokenizer::BBPETokenizer::load_with_options(
            filepath,
            LoadOptions {
                allow_legacy,
                validate,
            },
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // m8：回退警告必须 Python 侧可见。Rust 侧只发 `tracing::warn`，而 Python
        // 默认没有安装 tracing subscriber（除非显式调用 init_rust_logger），
        // 用户会在"用错 pattern 分词"的情况下毫无察觉 —— 这正是契约要消灭的 Q2。
        // 这里用 `warnings.warn` 兜底，保证任何情况下都会出现在 stderr。
        if inner.state.legacy_pattern_fallback {
            let msg = format!(
                "bbpe_tokenizer.BBPETokenizer: 模型 {filepath} 缺少 pre_tokenizer_pattern\
                 （v0 旧格式），已回退 DEFAULT_PATTERN。若该模型训练时使用了自定义 pattern，\
                 编码结果将是错的 —— 请用 `bbpe inspect --migrate <in> <out> --pattern <regex>` 迁移。"
            );
            warn_python(&msg);
        }
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    // R4-4 / R4-16：`encode` 改用**严格**路径（预分词失败 ⟹ Python 异常，
    // 而不是返回不完整编码），并在 Rust 计算期间释放 GIL。
    //
    // 此前签名不返回 Result，自定义 pattern 未覆盖全部文本时调用方会收到
    // 部分 token（`decode(encode(text)) != text`），只有 stderr 告警 ——
    // 与"错误映射为 Python 异常"的声明不符。
    #[pyo3(signature = (text, allowed_special=None))]
    pub fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<PyEncoding> {
        // R6-1：owned 数据 + Arc 模型，供长文本路径搬进 worker。
        let owned_text = text.to_string();
        let owned_special = allowed_special.unwrap_or_default();
        let inner = self.inner.clone();
        let rust_encoding =
            run_inference_interruptible(py, text.len() >= INFERENCE_INTERRUPT_BYTES, move || {
                // A3：与 encode_parallel 共用同一判定逻辑，不重复实现
                let (set, allow_all) = resolve_allowed_special(Some(&owned_special));
                inner
                    .encode_checked(&owned_text, &set, allow_all)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        Ok(rust_encoding.into())
    }

    /// 极速编码路径：不解析特殊字符，不追踪 offsets，直接返回 ID 数组
    ///
    /// R4-4 收口：与 `encode` 同口径走**严格**路径 —— 预分词失败抛
    /// `RuntimeError`，而不是返回空/不完整结果且仅 stderr 告警（机器调用方
    /// 此前无法感知）。Rust 侧非严格快速路径 `encode_ordinary` 的“失败返回
    /// 不完整结果”语义仅限显式选用它的 Rust 调用方。
    #[pyo3(signature = (text))]
    pub fn encode_ordinary(&self, py: Python<'_>, text: &str) -> PyResult<Vec<i32>> {
        let owned_text = text.to_string();
        let inner = self.inner.clone();
        run_inference_interruptible(py, text.len() >= INFERENCE_INTERRUPT_BYTES, move || {
            inner
                .encode_ordinary_checked(&owned_text)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// P3 新增：多线程全功能并行编码 (专门用于超长文本)
    #[pyo3(signature = (text, allowed_special=None, num_threads=None))]
    pub fn encode_parallel(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: Option<Vec<String>>,
        num_threads: Option<usize>,
    ) -> PyResult<PyEncoding> {
        let owned_text = text.to_string();
        let owned_special = allowed_special.unwrap_or_default();
        let inner = self.inner.clone();
        let rust_encoding =
            run_inference_interruptible(py, text.len() >= INFERENCE_INTERRUPT_BYTES, move || {
                // A3：与 encode 共用判定，避免 m7 那类分叉回归
                let (set, allow_all) = resolve_allowed_special(Some(&owned_special));
                inner
                    .encode_parallel_checked(&owned_text, &set, allow_all, num_threads)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        Ok(rust_encoding.into())
    }

    /// P3 新增：多线程极速裸 ID 编码 (专门用于超长文本)
    ///
    /// R4-4 收口：与 `encode_parallel` 同口径走严格路径（预分词失败抛异常）。
    #[pyo3(signature = (text, num_threads=None))]
    pub fn encode_ordinary_parallel(
        &self,
        py: Python<'_>,
        text: &str,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<i32>> {
        let owned_text = text.to_string();
        let inner = self.inner.clone();
        run_inference_interruptible(py, text.len() >= INFERENCE_INTERRUPT_BYTES, move || {
            inner
                .encode_ordinary_parallel_checked(&owned_text, num_threads)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    pub fn decode(&self, py: Python<'_>, ids: Vec<i32>) -> PyResult<String> {
        let inner = self.inner.clone();
        run_inference_interruptible(py, ids.len() >= INFERENCE_INTERRUPT_IDS, move || {
            inner
                .decode(&ids)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        })
    }

    /// 获取某个 Token ID 的 (provides, hangs) 属性
    pub fn get_utf8_token_props(&self, token_id: i32) -> PyResult<(usize, usize)> {
        if token_id < 0 {
            return Ok((0, 0));
        }
        let idx = token_id as usize;

        // 极限 O(1) 数组查询，取代了昂贵的 HashMap
        if idx < self.inner.fsm_props_array.len() {
            let props = self.inner.fsm_props_array[idx];
            Ok((props.provides, props.hangs))
        } else {
            // 对于越界或其他未知的 token，默认为最安全的普通文本属性
            Ok((0, 0))
        }
    }

    /// 获取在当前 pending_bytes 状态下，词表中所有【合法】的 Token IDs 列表
    pub fn get_valid_continuations(&self, pending_bytes: usize) -> PyResult<Vec<i32>> {
        match pending_bytes {
            0 => Ok(self.inner.valid_for_pending_0.clone()),
            1 => Ok(self.inner.valid_for_pending_1.clone()),
            2 => Ok(self.inner.valid_for_pending_2.clone()),
            3 => Ok(self.inner.valid_for_pending_3.clone()),
            // 理论上不可能超过 3，如果出现异常，返回空数组阻断生成
            _ => Ok(vec![]),
        }
    }

    /// 【极速探测】该 token id 是否为特殊控制 token（O(1) 查表）。
    ///
    /// 此前只有 Rust 侧可达：Python 想判断"这个 id 要不要计 loss"只能自己
    /// 再从模型里读一遍 special_tokens 做集合，两处口径迟早分叉。
    pub fn is_special_id(&self, token_id: i32) -> bool {
        self.inner.is_special_id(token_id)
    }

    /// 【截断防线】在该 token 之后截断是否安全（不会把多字节字符切成半个）。
    ///
    /// 长序列 Packing / `max_length` 截断前逐个检查该 id 的 `hangs` 属性：
    /// 返回 `False` 表示此处截断会产生乱码，应继续往后找安全位置。
    pub fn is_safe_to_truncate_after(&self, token_id: i32) -> bool {
        self.inner.is_safe_to_truncate_after(token_id)
    }

    /// 批量极速编码：返回与 `texts` 等长的 id 数组列表（Rayon 并行）。
    ///
    /// 与逐条 `encode_ordinary` 相比少了每条的 Python↔Rust 往返；语义同
    /// `encode_ordinary`（**严格**：任一条预分词失败即整体抛 `RuntimeError`，
    /// 而不是返回部分结果）。
    pub fn encode_ordinary_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
    ) -> PyResult<Vec<Vec<i32>>> {
        let total: usize = texts.iter().map(|t| t.len()).sum();
        let inner = self.inner.clone();
        run_inference_interruptible(py, total >= INFERENCE_INTERRUPT_BYTES, move || {
            use rayon::prelude::*;
            texts
                .par_iter()
                .map(|t| inner.encode_ordinary_checked(t))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// 返回按 merges 数 `k` 截断的**新**分词器（等价于 CLI `bbpe cut`，零重训）。
    ///
    /// 口径提醒：`k` 是 **merges 条数**，总词表 = `256 + k + len(special_tokens)`。
    /// CLI 的 `--vocab` 是总词表口径（`k = vocab - 256`），两者换算见
    /// `cli::util::vocab_to_k`。`k` 超过源模型规模时自动钳制。
    pub fn cut_to_merges(&self, k: usize) -> PyResult<Self> {
        let inner = self
            .inner
            .cut_to_merges(k)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// 返回改写预分词 pattern 的**新**分词器（`self` 不变）。
    ///
    /// 用途：v0 旧模型迁移 / 修正训练时选错的 pattern（`inspect --migrate
    /// --pattern` 的 Python 等价物）。pattern 是否**全文覆盖**需自行用
    /// `PreTokenizer.pre_tokenize_checked` 验证 —— 编码路径会在遇到未覆盖
    /// 文本时抛错，而不是静默丢字符。
    pub fn rewrite_pattern(&self, pattern: &str) -> PyResult<Self> {
        let inner = self
            .inner
            .with_pattern(pattern)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// 模型自洽性校验（V1–V7），不通过时抛 `ValueError`。
    ///
    /// 等价于 CLI `bbpe inspect --validate-only`。
    pub fn validate(&self) -> PyResult<()> {
        self.inner
            .state
            .validate()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // ---------- 模型读取面（此前 Python 完全不可见） ----------

    /// 词表项数（`256` 基础字节 + merges + special_tokens）。
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.state.vocab.len()
    }

    /// 本模型文件里**实际生效**的预分词正则（不是训练配置的推测值）。
    #[getter]
    fn pattern(&self) -> String {
        self.inner.state.pre_tokenizer_pattern.clone()
    }

    /// 文件声明的 `format`（v0 旧格式 ⟹ `None`）。
    #[getter]
    fn format(&self) -> Option<String> {
        self.inner.state.format.clone()
    }

    /// 文件声明的 `format_version`（v0 旧格式 ⟹ `None`）。
    #[getter]
    fn format_version(&self) -> Option<u32> {
        self.inner.state.format_version
    }

    /// `meta` 块（corpus / sampling / train / eval）转 dict；文件无 meta 时返回 `None`。
    ///
    /// 此前 Python 只能靠 CLI `inspect` 看这些信息 —— 而"训练用了多少行、
    /// 跳过多少、从哪个规模截断的"恰恰是复现与审计最需要盯的字段。
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(meta) = self.inner.state.meta.as_ref() else {
            return Ok(None);
        };
        let value =
            serde_json::to_value(meta).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Some(json_to_py(py, &value)?))
    }

    #[pyo3(signature = (filepath, no_meta=false))]
    pub fn save(&self, filepath: &str, no_meta: bool) -> PyResult<()> {
        self.inner
            .save_with_options(filepath, SaveOptions { no_meta })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// 与另一个**已加载**模型对照合并序列（等价 CLI `bbpe compare`，但不读盘）。
    ///
    /// 返回结构化报告：`first_divergence`（0-based，`None` = 前 `compared` 项
    /// 逐位完全一致）、`positional`、`at`（各分段的三项指标）。比较在
    /// **字节空间**进行 —— id 空间不可比（`new_id = 256 + i` 会因次序分叉而整体
    /// 错位，产出 ~6% 的假差异），口径说明见 `crate::compare`。
    ///
    /// 典型用途：验证 `cut_to_merges(k)` 是纯截断 ——
    /// `tk.compare(tk.cut_to_merges(k))["first_divergence"]` 必为 `None`。
    #[pyo3(signature = (other, *, at=None))]
    pub fn compare(
        &self,
        py: Python<'_>,
        other: PyRef<'_, Self>,
        at: Option<Vec<usize>>,
    ) -> PyResult<Py<PyDict>> {
        let at = at.unwrap_or_else(|| DEFAULT_AT.to_vec());
        let report = compare_states(&self.inner.state, &other.inner.state, &at);
        report_to_py(py, &report)
    }
}

/// A3 / m7：把 Python 侧的 `allowed_special` 解析**收敛到唯一一处**。
///
/// 背景：m7 修掉了 `encode` 里过窄的判定（`vec.len() == 1 && vec[0] == "all"`
/// 只在"恰好一个元素且为 all"时生效，`["<pad>", "all"]` 会静默退化成字面量
/// 集合 —— 用户以为放行了全部 special，实际一个都没放行，且无任何提示），
/// **但漏改了同文件的 `encode_parallel`**。于是同一输入在两个 API 下给出
/// 不同的 ids，防注入语义在并行路径被静默降级。
///
/// 根因是同一段逻辑被复制了两份。改为抽出本函数供两处共用，从结构上杜绝
/// 再次分叉 —— 这与 `tokenizer.rs` 把三份 BPE 合并循环收成 `bpe_merge_core`
/// （m2）是同一种修法：**消除复制，而非就地修补**。
///
/// 返回 `(允许集合, 是否放行全部)`。
fn resolve_allowed_special(vec: Option<&Vec<String>>) -> (HashSet<&str>, bool) {
    let mut allow_all = false;
    let mut allowed_set = HashSet::new();

    if let Some(vec) = vec {
        // m7：任一项为 "all" 即放行全部（与直觉一致），而非要求"恰好一个元素"。
        if vec.iter().any(|s| s == "all") {
            allow_all = true;
        } else {
            for s in vec {
                allowed_set.insert(s.as_str());
            }
        }
    }
    (allowed_set, allow_all)
}

/// m8：向 Python 侧发出告警（`warnings.warn` → stderr）。
///
/// 为什么需要它：Rust 侧的重要提示走 `tracing::warn!`，而 Python 进程默认
/// **没有**安装 tracing subscriber（需显式 `init_rust_logger()`），于是这些
/// 提示全部消失。绑定层的关键告警必须有不依赖 subscriber 的通道。
/// `warnings` 模块缺失时退化到 `builtins.print`，保证绝不静默。
fn warn_python(msg: &str) {
    Python::attach(|py| {
        let warned = py
            .import("warnings")
            .and_then(|m| m.getattr("warn"))
            .and_then(|w| w.call1((msg,)))
            .is_ok();
        if !warned {
            let _ = py
                .import("builtins")
                .and_then(|m| m.getattr("print"))
                .and_then(|p| p.call1((msg,)));
        }
    })
}

/// 训练/采样统计 → Python dict。
///
/// `skipped_lines`、`io_errors` 是"语料被静默改动/截断"的唯一信号，
/// 必须可被机器断言，而不是只出现在 warning 文本里。
fn build_stats_dict(py: Python<'_>, stats: &CorpusStats) -> PyResult<Py<PyDict>> {
    let (total_lines, total_bytes, skipped_lines, replacement_chars) = stats.snapshot();
    let d = PyDict::new(py);
    d.set_item("total_lines", total_lines)?;
    d.set_item("total_bytes", total_bytes)?;
    d.set_item("skipped_lines", skipped_lines)?;
    d.set_item("replacement_chars", replacement_chars)?;
    d.set_item("io_errors", stats.io_errors())?;
    Ok(d.unbind())
}

/// `serde_json::Value` → Python 对象（dict / list / str / int / float / bool / None）。
///
/// 只用于把 `meta` 这类纯记录块交给 Python，故不做自定义类型映射。
fn json_to_py<'py>(py: Python<'py>, v: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    use serde_json::Value;
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any()
            }
        }
        Value::String(s) => s.as_str().into_pyobject(py)?.into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, val) in map {
                dict.set_item(k, json_to_py(py, val)?)?;
            }
            dict.into_any()
        }
    })
}

/// `CompareReport` → Python dict（走 serde，避免逐字段手抄导致漏字段）。
fn report_to_py(py: Python<'_>, report: &CompareReport) -> PyResult<Py<PyDict>> {
    let value = serde_json::to_value(report).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let obj = json_to_py(py, &value)?;
    let dict = obj
        .cast_into::<PyDict>()
        .map_err(|_| PyRuntimeError::new_err("对照报告序列化失败（内部错误）"))?;
    Ok(dict.unbind())
}

/// 对照两个模型文件的合并序列（等价 CLI `bbpe compare`）。
///
/// 回答"采样率 / 语料规模是否足够"：返回 `first_divergence`（0-based，
/// `None` = 前 `compared` 项逐位完全一致）与各分段的三项指标，可直接用于
/// `assert` / 画图 / 入库 —— CLI 只能打印文本表格。
///
/// 只加载 `BpeState`（不建 FSM/Aho-Corasick 缓存），比构造两个
/// `BBPETokenizer` 更省内存。
#[pyfunction]
#[pyo3(
    name = "compare_models",
    signature = (path_a, path_b, *, at=None, allow_legacy=false)
)]
fn py_compare_models(
    py: Python<'_>,
    path_a: &str,
    path_b: &str,
    at: Option<Vec<usize>>,
    allow_legacy: bool,
) -> PyResult<Py<PyDict>> {
    let at = at.unwrap_or_else(|| DEFAULT_AT.to_vec());
    let opts = LoadOptions {
        allow_legacy,
        validate: true,
    };
    let a = path_a.to_string();
    let b = path_b.to_string();
    let report = py
        .detach(move || -> Result<CompareReport, BbpeError> {
            let sa = BpeState::load_with_options(&a, opts.clone())?;
            let sb = BpeState::load_with_options(&b, opts)?;
            Ok(compare_states(&sa, &sb, &at))
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let dict = report_to_py(py, &report)?;
    let bound = dict.bind(py);
    bound.set_item("a_path", path_a)?;
    bound.set_item("b_path", path_b)?;
    Ok(dict)
}

/// 单独采一次样（等价于 CLI `bbpe sample`）。
///
/// 何时需要它：要把同一份样本训多个模型，或做"rate₁ ⊆ rate₂"的嵌套对照。
/// 日常训练不必用 —— `BBPETrainer.train(sample_rate=...)` 已内建采样。
///
/// 前置校验与 CLI **逐条一致**（R4-22 输入/输出别名防护、区间检查、
/// `heldout_rate > 0` 必须同时给 `heldout_out`），因为这些条件一旦漏检，
/// 后果是"边读边写同一个文件"或"命中行被静默丢弃"，事后无法补救。
#[pyfunction]
#[pyo3(
    name = "sample_lines",
    signature = (input_path, train_out, *, rate, seed=42, max_bytes=None,
                 heldout_rate=0.0, heldout_seed=43, heldout_out=None)
)]
fn py_sample_lines(
    py: Python<'_>,
    input_path: &str,
    train_out: &str,
    rate: f64,
    seed: u64,
    max_bytes: Option<u64>,
    heldout_rate: f64,
    heldout_seed: u64,
    heldout_out: Option<&str>,
) -> PyResult<Py<PyDict>> {
    if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
        return Err(PyValueError::new_err("rate 必须在 [0, 1]"));
    }
    if !heldout_rate.is_finite() || !(0.0..=1.0).contains(&heldout_rate) {
        return Err(PyValueError::new_err("heldout_rate 必须在 [0, 1]"));
    }
    if heldout_rate > 0.0 && heldout_out.is_none() {
        return Err(PyValueError::new_err(
            "heldout_rate > 0 时必须提供 heldout_out（否则命中行会被静默丢弃）",
        ));
    }
    if same_path(input_path, train_out) {
        return Err(PyValueError::new_err(
            "input_path 与 train_out 指向同一个文件：采样输出会覆盖原始语料",
        ));
    }
    if let Some(h) = heldout_out {
        if same_path(input_path, h) {
            return Err(PyValueError::new_err(
                "input_path 与 heldout_out 指向同一个文件：held-out 输出会覆盖原始语料",
            ));
        }
        if same_path(h, train_out) {
            return Err(PyValueError::new_err(
                "heldout_out 与 train_out 指向同一个文件：两者内容不同，共路径会互相覆盖",
            ));
        }
    }

    // 采样是纯 IO + CPU，全程放开 GIL（大语料下 Python 线程不该被卡住）。
    let owned_input = input_path.to_string();
    let owned_train = train_out.to_string();
    let owned_held = heldout_out.map(|s| s.to_string());
    let stats = py
        .detach(move || -> std::io::Result<_> {
            ensure_parent(&owned_train)?;
            let fin = File::open(&owned_input)?;
            let mut reader = BufReader::with_capacity(1 << 20, fin);
            let mut train_w = BufWriter::with_capacity(1 << 20, File::create(&owned_train)?);
            let mut held_w = match owned_held.as_deref() {
                Some(p) => {
                    ensure_parent(p)?;
                    Some(BufWriter::with_capacity(1 << 20, File::create(p)?))
                }
                None => None,
            };
            let s = sample_lines(
                &mut reader,
                &mut train_w,
                held_w.as_mut(),
                SampleParams {
                    rate,
                    seed,
                    heldout_rate,
                    heldout_seed,
                    max_bytes: max_bytes.unwrap_or(u64::MAX),
                },
            )?;
            train_w.flush()?;
            if let Some(w) = held_w.as_mut() {
                w.flush()?;
            }
            Ok(s)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("total_lines", stats.total_lines)?;
    d.set_item("total_bytes", stats.total_bytes)?;
    d.set_item("train_lines", stats.train_lines)?;
    d.set_item("train_bytes", stats.train_bytes)?;
    d.set_item("held_lines", stats.held_lines)?;
    d.set_item("held_bytes", stats.held_bytes)?;
    Ok(d.unbind())
}

/// 暴露一个初始化 Rust tracing 日志的函数，在 Python 中调用后才能在控制台看到 Rust 的日志
#[pyfunction]
fn init_rust_logger() {
    let _ = tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .try_init(); // 使用 try_init 避免重复调用时 panic
}

#[pymodule]
fn bbpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPatternFlavor>()?;
    m.add_class::<PyTrainerConfig>()?;
    m.add_class::<PyRuntimeConfig>()?;
    m.add_class::<PyBBPETrainer>()?;
    m.add_class::<PyBBPETokenizer>()?;
    m.add_class::<PyTextFileReader>()?; // 注册新增的类
    m.add_class::<PyJsonlFileReader>()?; // 注册新增的类
    m.add_class::<PyLineIter>()?;
    // 预分词器：pattern 预览 / 覆盖校验（此前只有 Rust 可达）
    m.add_class::<PyPreTokenizer>()?;
    // R6-1 P2：取消令牌与程序化取消异常
    m.add_class::<PyCancelToken>()?;
    m.add("TrainingCancelled", m.py().get_type::<TrainingCancelled>())?;
    m.add_function(wrap_pyfunction!(init_rust_logger, m)?)?;
    // 单独采样（等价 CLI `bbpe sample`）
    m.add_function(wrap_pyfunction!(py_sample_lines, m)?)?;
    // 合并序列对照（等价 CLI `bbpe compare`）
    m.add_function(wrap_pyfunction!(py_compare_models, m)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::resolve_allowed_special;

    /// A3：`encode` 与 `encode_parallel` 必须共用同一套 `allowed_special` 语义。
    ///
    /// 回归背景：m7 修掉了 `encode` 里过窄的判定，却**漏改** `encode_parallel`，
    /// 于是 `["<pad>", "all"]` 在串行路径放行全部 special、在并行路径一个都
    /// 不放行 —— 同一输入两个 API 给出不同 ids，防注入语义被静默降级。
    /// 本测试把判定收敛到 `resolve_allowed_special` 后，用穷举钉住语义。
    #[test]
    fn a3_all_anywhere_allows_every_special() {
        let cases: Vec<Vec<&str>> = vec![
            vec!["all"],
            vec!["all", "all"],
            vec!["<pad>", "all"],
            vec!["all", "<pad>"],
            vec!["<pad>", "<eos>", "all"],
        ];
        for c in cases {
            let v: Vec<String> = c.iter().map(|s| s.to_string()).collect();
            let (set, allow_all) = resolve_allowed_special(Some(&v));
            assert!(allow_all, "{c:?} 含 all 时必须放行全部");
            assert!(
                set.is_empty(),
                "{c:?} 走 allow_all 分支时不应同时产生字面量集合（否则语义含混）"
            );
        }
    }

    #[test]
    fn a3_without_all_uses_literal_set() {
        let v: Vec<String> = vec!["<|im_start|>".into(), "<|im_end|>".into()];
        let (set, allow_all) = resolve_allowed_special(Some(&v));
        assert!(!allow_all);
        assert_eq!(set.len(), 2);
        assert!(set.contains("<|im_start|>"));
        assert!(set.contains("<|im_end|>"));
    }

    #[test]
    fn a3_empty_and_none_allow_no_special() {
        let empty: Vec<String> = vec![];
        let (set, allow_all) = resolve_allowed_special(Some(&empty));
        assert!(!allow_all && set.is_empty(), "空列表不应放行任何 special");

        let (set2, allow_all2) = resolve_allowed_special(None);
        assert!(!allow_all2 && set2.is_empty(), "None 应等价于空列表");
    }

    /// 字面量 `"all"` 只有在**整项就是** "all" 时才触发全放行；
    /// 形如 `"allowed"` / `"ALL"` 的特殊 token 名不应被误判。
    #[test]
    fn a3_only_exact_all_token_matches() {
        for name in ["allowed", "ALL", "All", "all_special", "alll"] {
            let v = vec![name.to_string()];
            let (set, allow_all) = resolve_allowed_special(Some(&v));
            assert!(!allow_all, "{name:?} 不应被当作 all");
            assert!(set.contains(name), "{name:?} 应作为字面量 special 放行");
        }
    }
}

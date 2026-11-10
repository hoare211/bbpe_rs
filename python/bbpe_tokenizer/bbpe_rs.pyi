"""bbpe_tokenizer.bbpe_rs 的类型存根。

与 `src/bindings/python.rs` 的 `#[pyclass]` / `#[pymethods]` **一一对应** ——
本文件是 Rust 公开能力的镜像：Rust 侧新增/修改任何 `#[pymethods]` 都必须同步
到这里，否则类型检查器会认为"Python 没有这个能力"（2026-11-08 接口核对中，
`RuntimeConfig.max_chunk_bytes` 就是这样缺了整整一轮）。

方法签名来自 PyO3 生成的 `__text_signature__`，`train()` 的关键字参数
顺序与默认值以 Rust 侧 `#[pyo3(signature = ...)]` 为准。
"""

from typing import Any, Iterator, Optional, Sequence

class PatternFlavor:
    @staticmethod
    def gpt4() -> PatternFlavor: ...
    @staticmethod
    def llama3() -> PatternFlavor: ...
    @staticmethod
    def custom(pattern: str) -> PatternFlavor: ...

class Encoding:
    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]

class PreTokenizer:
    """预分词器：pattern 的直接载体，可在编码前验证切分口径。"""

    def __init__(
        self, pattern: str, max_chunk_bytes: Optional[int] = None
    ) -> None: ...
    def pre_tokenize(self, text: str) -> list[tuple[bytes, int]]:
        """**非严格**预览：未被 pattern 覆盖的间隙会被静默丢弃。"""

    def pre_tokenize_checked(self, text: str) -> list[tuple[bytes, int]]:
        """严格预览：pattern 未覆盖全部文本时抛 `RuntimeError`。"""

    @property
    def pattern(self) -> str: ...
    @property
    def max_chunk_bytes(self) -> int: ...

class TrainerConfig:
    def __init__(self, vocab_size: int) -> None: ...
    def with_pattern(self, flavor: PatternFlavor) -> TrainerConfig: ...
    def with_min_frequency(self, freq: int) -> TrainerConfig: ...
    def with_special_tokens(self, tokens: Sequence[str]) -> TrainerConfig: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def min_frequency(self) -> int: ...
    @property
    def special_tokens(self) -> list[str]: ...
    @property
    def pattern(self) -> str: ...

class RuntimeConfig:
    def __init__(self) -> None: ...
    def with_batch_size(self, size: int) -> RuntimeConfig: ...
    def with_num_shards(self, shards: int) -> RuntimeConfig: ...
    def with_parallel_index(self, parallel: bool) -> RuntimeConfig: ...
    def with_max_chunk_bytes(self, bytes: int) -> RuntimeConfig: ...
    @property
    def batch_size(self) -> int: ...
    @property
    def num_shards(self) -> int: ...
    @property
    def parallel_index(self) -> bool: ...
    @property
    def max_chunk_bytes(self) -> int: ...

class LineIterator(Iterator[str]):
    def __iter__(self) -> LineIterator: ...
    def __next__(self) -> str: ...

class TextFileReader:
    def __init__(self, filepath: str) -> None: ...
    def iter_lines(self) -> LineIterator: ...
    @property
    def filepath(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def file_size(self) -> int: ...

class JsonlFileReader:
    # 该类型没有 #[new]，只能用下面的静态构造器创建
    @staticmethod
    def static_text(filepath: str) -> JsonlFileReader: ...
    @staticmethod
    def static_conversations(
        filepath: str, separator: str = "\n"
    ) -> JsonlFileReader: ...
    @staticmethod
    def dynamic(
        filepath: str, schema: dict[str, tuple[str, str]]
    ) -> JsonlFileReader: ...
    def iter_lines(self) -> LineIterator: ...
    @property
    def filepath(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def file_size(self) -> int: ...

class CancellationToken:
    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    @property
    def cancelled(self) -> bool: ...
    @property
    def reason(self) -> str: ...
    @property
    def merges_done(self) -> int: ...

class TrainingCancelled(RuntimeError): ...

class BBPETrainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
        runtime_config: Optional[RuntimeConfig] = None,
    ) -> None: ...
    def train(
        self,
        reader: TextFileReader | JsonlFileReader,
        *,
        cancel_token: Optional[CancellationToken] = None,
        timeout: Optional[float] = None,
        sample_rate: Optional[float] = None,
        sample_seed: int = 42,
        max_bytes: Optional[int] = None,
        oom_precheck: Optional[bool] = None,
        heldout_rate: float = 0.0,
        heldout_seed: int = 43,
        heldout_out: Optional[str] = None,
        sample_out: Optional[str] = None,
    ) -> dict[str, int]:
        """训练。返回语料统计：

        `total_lines` / `total_bytes` / `skipped_lines` / `replacement_chars`
        / `io_errors`。`io_errors > 0` 表示语料读取**中途被截断**（不是读完）。
        """

    def set_pre_tokenizer(self, pre_tokenizer: PreTokenizer) -> None: ...
    def save(self, filepath: str, no_meta: bool = False) -> None: ...
    def wait_cleanup(self) -> None: ...
    def reset(self) -> None: ...
    def is_trained(self) -> bool: ...

class BBPETokenizer:
    def __init__(
        self, filepath: str, allow_legacy: bool = False, validate: bool = True
    ) -> None: ...
    def encode(
        self, text: str, allowed_special: Optional[Sequence[str]] = None
    ) -> Encoding: ...
    def encode_ordinary(self, text: str) -> list[int]: ...
    def encode_ordinary_batch(self, texts: Sequence[str]) -> list[list[int]]: ...
    def encode_parallel(
        self,
        text: str,
        allowed_special: Optional[Sequence[str]] = None,
        num_threads: Optional[int] = None,
    ) -> Encoding: ...
    def encode_ordinary_parallel(
        self, text: str, num_threads: Optional[int] = None
    ) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def get_utf8_token_props(self, token_id: int) -> tuple[int, int]: ...
    def get_valid_continuations(self, pending_bytes: int) -> list[int]: ...
    def is_special_id(self, token_id: int) -> bool: ...
    def is_safe_to_truncate_after(self, token_id: int) -> bool: ...
    def cut_to_merges(self, k: int) -> BBPETokenizer:
        """按 merges 数 k 截断（总词表 = 256 + k + len(special_tokens)）。"""

    def rewrite_pattern(self, pattern: str) -> BBPETokenizer: ...
    def compare(
        self, other: BBPETokenizer, *, at: Optional[Sequence[int]] = None
    ) -> dict[str, Any]:
        """与另一个已加载模型对照合并序列（字节空间，等价 CLI `bbpe compare`）。"""

    def validate(self) -> None: ...
    def meta(self) -> Optional[dict[str, Any]]: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def pattern(self) -> str: ...
    @property
    def format(self) -> Optional[str]: ...
    @property
    def format_version(self) -> Optional[int]: ...
    def save(self, filepath: str, no_meta: bool = False) -> None: ...

def compare_models(
    path_a: str,
    path_b: str,
    *,
    at: Optional[Sequence[int]] = None,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """对照两个模型文件的合并序列（等价 CLI `bbpe compare`）。

    返回 `merges_a` / `merges_b` / `compared` / `first_divergence`
    （0-based，`None` = 前 `compared` 项逐位完全一致）/ `divergence` /
    `positional` / `at`（各分段的三项指标）/ `a_path` / `b_path`。

    `positional["ratio"]` 在 `compared == 0`（任一侧 merges 为 0，无从比较）时
    为 `None` —— 不是 0 也不是 1，调用方必须显式处理。

    比较在**字节空间**进行；`at[*].id_space_ratio` 仅作对照（编号错位效应），
    不可作为结论依据。
    """

def sample_lines(
    input_path: str,
    train_out: str,
    *,
    rate: float,
    seed: int = 42,
    max_bytes: Optional[int] = None,
    heldout_rate: float = 0.0,
    heldout_seed: int = 43,
    heldout_out: Optional[str] = None,
) -> dict[str, int]:
    """单独采一次样（等价 CLI `bbpe sample`）。

    返回 `total_lines` / `total_bytes` / `train_lines` / `train_bytes`
    / `held_lines` / `held_bytes`。
    """

def init_rust_logger() -> None: ...

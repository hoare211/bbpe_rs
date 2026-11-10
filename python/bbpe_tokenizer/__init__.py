# python/bbpe_tokenizer/__init__.py
"""bbpe_tokenizer —— Rust 核心 + Python 薄封装。

注意：`UTF8StrictLogitsProcessor` 走**惰性导入**（PEP 562 的模块级
`__getattr__`）。原因：它依赖 `torch` + `transformers`，而这两个是重依赖 ——
此前 `__init__.py` 无条件 `from .logits_processor import ...`，导致
**只想用分词器/训练器的人也被迫安装 torch**，否则
`import bbpe_tokenizer` 直接 ModuleNotFoundError。

惰性导入后：
- `from bbpe_tokenizer import BBTokenizer, BBPETrainer` —— 无需 torch；
- `from bbpe_tokenizer import UTF8StrictLogitsProcessor` —— 仍然可用，
  只在**真正取用时**才 import torch，此时报出的错误也更贴近使用场景。

打包侧同步：torch / transformers 已移入 `pyproject.toml` 的
`[project.optional-dependencies].logits`（B5），默认安装不再拉它们；
需要该处理器时 `pip install bbpe_tokenizer[logits]`。
"""

from collections.abc import Iterator
from contextlib import contextmanager

from .config import TokenizerGlobalConfig
from .core import BBTokenizer

# 【Rust 扩展】：TrainerConfig / RuntimeConfig / Reader 等
from .bbpe_rs import (
    BBPETrainer,
    TrainerConfig,
    RuntimeConfig,
    TextFileReader,
    JsonlFileReader,
    init_rust_logger,
    PatternFlavor,
    CancellationToken,
    TrainingCancelled,
)

__all__ = [
    # 高级业务层
    "TokenizerGlobalConfig",
    "BBTokenizer",
    "UTF8StrictLogitsProcessor",

    # 数据集读取与处理基建
    "TextFileReader",
    "JsonlFileReader",

    # 训练引擎与配置
    "BBPETrainer",
    "TrainerConfig",
    "RuntimeConfig",
    "init_rust_logger",
    "PatternFlavor",

    # 训练取消（R6-1 P2）
    "CancellationToken",
    "TrainingCancelled",
    "interruptible",
]

@contextmanager
def interruptible() -> Iterator[CancellationToken]:
    """把 Ctrl+C 接到一个 :class:`CancellationToken`，供后台线程里的训练使用。

    主线程调用 ``trainer.train()`` 时 Ctrl+C 已自动生效，无需本函数。
    只有当训练跑在**别的 Python 线程**时才需要它 —— 因为 CPython 的
    ``PyErr_CheckSignals`` 只在主线程调用 Python 信号处理器。

    用法::

        with bbpe_tokenizer.interruptible() as tok:
            t = threading.Thread(
                target=lambda: trainer.train(reader, cancel_token=tok)
            )
            t.start()
            t.join()

    handler 先 ``tok.cancel()`` 让 worker 立即停下，再链式调用原 handler
    （默认会抛 ``KeyboardInterrupt``），因此主线程拿到异常时后台训练
    已在收尾，不会出现进程 shutdown 卡住。退出 ``with`` 时恢复原 handler。

    非主线程调用时（``signal.signal`` 只能在主线程安装）**不报错**：发一条
    ``RuntimeWarning`` 后退化成只返回令牌，调用方仍可用 ``tok.cancel()``
    或 ``trainer.train(..., timeout=)`` 取消训练。
    经此路径取消时异常的 ``reason`` 为 ``"user"``。
    """
    import signal
    import threading
    import warnings

    if threading.current_thread() is not threading.main_thread():
        warnings.warn(
            "interruptible() 在非主线程调用：不安装 SIGINT handler，仅返回 "
            "CancellationToken；请自行调用 tok.cancel() 或设置 timeout=。",
            RuntimeWarning,
            stacklevel=2,
        )
        yield CancellationToken()
        return

    token = CancellationToken()
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        token.cancel()
        if callable(previous):
            return previous(signum, frame)
        if previous == signal.SIG_DFL:
            raise KeyboardInterrupt
        return None

    signal.signal(signal.SIGINT, handler)
    try:
        yield token
    finally:
        signal.signal(signal.SIGINT, previous)


# 需要 torch 的可选组件，惰性解析（见模块 docstring）
_LAZY = {"UTF8StrictLogitsProcessor": ("logits_processor", "UTF8StrictLogitsProcessor")}


def __getattr__(name: str):
    """PEP 562：仅在真正取用惰性符号时才 import 其所在模块。"""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f".{target[0]}", __name__)
    value = getattr(mod, target[1])
    # 缓存到模块字典，后续取用不再走 __getattr__
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))

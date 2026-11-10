"""发布验收补充：取消 × 采样/推理的组合路径与资源回收压力。

这些用例在 release 构建下同样必须稳定 —— 因此全部改为**事件驱动**或
**子进程自触发信号**，不依赖「训练足够慢」这一 debug 特性。
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from bbpe_tokenizer import (
    BBPETrainer,
    CancellationToken,
    TextFileReader,
    TrainerConfig,
    TrainingCancelled,
)
from bbpe_tokenizer.bbpe_rs import BBPETokenizer

LINE = "中文语言文字处理与字节级 BPE 训练 abcabc 中文中文 0123456789 "
SAMPLE_GLOB = "/tmp/bbpe_py_sample_*"

CHILD_SAMPLE_SIGINT = r'''
import os, signal, sys, threading, time
import bbpe_tokenizer as b

corpus = sys.argv[1]
t = b.BBPETrainer(b.TrainerConfig(6000))


def fire():
    time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGINT)


threading.Thread(target=fire, daemon=True).start()
print("READY", flush=True)
try:
    with b.interruptible() as tok:
        t.train(b.TextFileReader(corpus), cancel_token=tok, sample_rate=0.5)
except KeyboardInterrupt:
    print("KBI", flush=True)
    print("TRAINED", t.is_trained(), flush=True)
else:
    print("DONE", flush=True)
'''

CHILD_DECODE_SIGINT = r'''
import os, signal, sys, threading, time
from bbpe_tokenizer.bbpe_rs import BBPETokenizer

model = sys.argv[1]
tok = BBPETokenizer(model)
ids = [259] * 20_000_000


def fire():
    time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGINT)


threading.Thread(target=fire, daemon=True).start()
print("READY", flush=True)
try:
    tok.decode(ids)
except KeyboardInterrupt:
    print("KBI", flush=True)
else:
    print("DONE", flush=True)
'''


def sample_files() -> set[str]:
    return set(glob.glob(SAMPLE_GLOB))


def _env() -> dict[str, str]:
    """子进程环境：仅当 python/ 内已有编译扩展时才注入源码目录。

    干净检出的 python/ 只有纯 Python 源码，扩展由 `maturin develop` 或安装
    wheel 提供；无条件注入会让子进程 import 失败（找不到 bbpe_rs）。
    """
    env = os.environ.copy()
    pkg = Path(__file__).resolve().parents[1] / "python" / "bbpe_tokenizer"
    if list(pkg.glob("bbpe_rs*.so")) or list(pkg.glob("bbpe_rs*.pyd")):
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(pkg.parent), env.get("PYTHONPATH", "")) if p
        )
    return env


class TestComboPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.corpus = root / "corpus.txt"
        cls.corpus.write_text(
            "".join(f"{LINE} 序号{i}\n" for i in range(200_000)), encoding="utf-8"
        )
        small = root / "small.txt"
        small.write_text((LINE + "\n") * 3000, encoding="utf-8")
        cls.model = str(root / "model.json")
        t = BBPETrainer(TrainerConfig(300))
        t.train(TextFileReader(str(small)))
        t.save(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_timeout_takes_effect_during_sampling(self):
        before = sample_files()
        trainer = BBPETrainer(TrainerConfig(4000))
        # 5ms：无论采样多快都落在「采样/预检/训练」窗口内，pump 全程覆盖
        with self.assertRaises(TrainingCancelled) as ctx:
            trainer.train(
                TextFileReader(str(self.corpus)), sample_rate=0.5, timeout=0.005
            )
        self.assertEqual(ctx.exception.reason, "timeout")
        trainer.wait_cleanup()
        self.assertEqual(sample_files(), before, "采样阶段取消后临时文件未清理")

    def test_interruptible_with_sampling_ctrl_c(self):
        before = sample_files()
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD_SAMPLE_SIGINT, str(self.corpus)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_env(),
        )
        try:
            self.assertEqual(proc.stdout.readline().strip(), "READY")
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail("采样阶段的 Ctrl+C 未在 30s 内生效")
        except BaseException:
            proc.kill()
            proc.communicate()
            raise

        self.assertIn("KBI", out, f"未收到 KeyboardInterrupt：{out}\n{err}")
        self.assertIn("TRAINED False", out, "取消后不应保留训练态")
        self.assertNotIn("DONE", out, "取消后不应完成训练")
        self.assertEqual(sample_files(), before, "临时样本文件残留")

    def test_long_ids_decode_interruptible(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD_DECODE_SIGINT, self.model],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_env(),
        )
        try:
            self.assertEqual(proc.stdout.readline().strip(), "READY")
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail("长 ids decode 未在 30s 内响应 SIGINT")
        except BaseException:
            proc.kill()
            proc.communicate()
            raise

        self.assertIn("KBI", out, f"decode 未收到 KeyboardInterrupt：{out}\n{err}")
        self.assertNotIn("DONE", out, "取消后 decode 不应完成")

    def test_ten_cancel_retrain_cycles_no_leak(self):
        before = sample_files()
        for i in range(10):
            tok = CancellationToken()
            tok.cancel()
            t = BBPETrainer(TrainerConfig(400))
            with self.assertRaises(TrainingCancelled):
                t.train(
                    TextFileReader(str(self.corpus)),
                    cancel_token=tok,
                    sample_rate=0.5,
                )
            t.wait_cleanup()
            t.reset()
            self.assertFalse(t.is_trained(), f"第 {i} 轮 reset 后仍残留训练态")
        self.assertEqual(sample_files(), before, "循环取消后临时文件累积")


if __name__ == "__main__":
    unittest.main()

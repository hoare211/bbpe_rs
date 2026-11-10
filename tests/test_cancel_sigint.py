"""R6-1：真实 SIGINT 必须能中断 Python 调用下的 Rust 训练。

只有进程级信号能证伪「信号根本没被处理」——`threading` 模拟替代不了，
因为 Python 层的定时器线程拿不到 GIL（P0 训练持 GIL；P1 虽释放 GIL，
但信号仍必须由主线程/主进程递送）。

P1 追加验收：异常 provenance、`wait_cleanup()`、取消后 `reset()` 重训。

用 unittest 编写，既能被 pytest 收集，也能在没有 pytest 的环境里直接
`python -m unittest discover -s tests -p 'test_cancel_sigint.py'` 运行。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHILD = r'''
import os
import signal
import sys
import threading
import time

from bbpe_tokenizer import BBPETrainer, TrainerConfig, TextFileReader

corpus, model, small, model2 = sys.argv[1:5]
t = BBPETrainer(TrainerConfig(__VOCAB__))


def fire():
    # 由子进程自己定时发信号：release 构建训练比 debug 快数倍，父进程
    # 固定 sleep 后再 kill 会落在训练结束之后。50ms 远小于任何构建下的
    # 训练时长；os.kill 与外部 kill 走同一条信号递送路径。
    time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGINT)


threading.Thread(target=fire, daemon=True).start()
print("READY", flush=True)
try:
    t.train(TextFileReader(corpus))
except KeyboardInterrupt as exc:
    print("INTERRUPTED", type(exc).__name__, flush=True)
    print(
        "PROVENANCE",
        getattr(exc, "merges_done", -1),
        getattr(exc, "reason", "?"),
        isinstance(getattr(exc, "elapsed_secs", None), float),
        isinstance(getattr(exc, "corpus_stats", None), dict),
        flush=True,
    )
    # 取消后的 state 已丢弃，save() 必须拒绝，磁盘上不能留下半截模型
    try:
        t.save(model)
    except RuntimeError:
        print("SAVE_REJECTED", flush=True)
    else:
        print("SAVE_ALLOWED", flush=True)
    t.wait_cleanup()
    t.reset()
    t.train(TextFileReader(small))
    t.save(model2)
    print("RESET_RETRAIN_OK", flush=True)
    sys.exit(0)
else:
    t.save(model)
    print("SAVED", flush=True)
    sys.exit(2)
'''

CHILD_CUSTOM_HANDLER = r'''
import os
import signal
import sys
import threading
import time

from bbpe_tokenizer import BBPETrainer, TrainerConfig, TextFileReader

class MyStop(Exception):
    pass

def handler(signum, frame):
    raise MyStop("来自自定义 handler 的取消")

signal.signal(signal.SIGINT, handler)
corpus = sys.argv[1]
t = BBPETrainer(TrainerConfig(__VOCAB__))


def fire():
    time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGINT)


threading.Thread(target=fire, daemon=True).start()
print("READY", flush=True)
t.train(TextFileReader(corpus))
print("SAVED", flush=True)
'''

LINE = "中文语言文字处理与字节级 BPE 分词器训练 abcabc 中文中文 0123456789 "


class _SigintBase(unittest.TestCase):
    vocab = 6000
    lines = 200000

    def _run_child(self, code: str):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        corpus = root / "corpus.txt"
        model = root / "model.json"
        small = root / "small.txt"
        model2 = root / "model2.json"
        # 必须是**真实多行**语料：整份拼成一行会让阶段 1 只有一次迭代，
        # 检查点密度与生产 JSONL 完全不同（P0 评审时踩过这个坑）。
        corpus.write_text((LINE + "\n") * self.lines, encoding="utf-8")
        small.write_text((LINE + "\n") * 2000, encoding="utf-8")

        env = os.environ.copy()
        # 仅当 python/ 内已有编译扩展时才注入源码目录（见 test_boundary_fixes.py 同款说明）
        pkg = Path(__file__).resolve().parents[1] / "python" / "bbpe_tokenizer"
        if list(pkg.glob("bbpe_rs*.so")) or list(pkg.glob("bbpe_rs*.pyd")):
            env["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(pkg.parent), env.get("PYTHONPATH", "")) if part
            )

        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                code.replace("__VOCAB__", str(self.vocab)),
                str(corpus),
                str(model),
                str(small),
                str(model2),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            ready = proc.stdout.readline().strip()
            self.assertEqual(ready, "READY", "子进程启动失败")
            # 信号由子进程自己定时发出（见 CHILD.fire）
            out, err = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            tmp.cleanup()
            self.fail("SIGINT 未在 20s 内中断训练（信号未被 Rust 侧检查）")
        except BaseException:
            proc.kill()
            proc.communicate()
            tmp.cleanup()
            raise

        result = (proc.returncode, out, err, model.exists(), model2.exists())
        tmp.cleanup()
        return result


class TestSigintCancellation(_SigintBase):
    def test_ctrl_c_interrupts_training_and_blocks_save(self):
        rc, out, err, model_exists, model2_exists = self._run_child(CHILD)
        self.assertEqual(rc, 0, f"子进程异常退出：{out}\n{err}")
        self.assertIn(
            "INTERRUPTED KeyboardInterrupt",
            out,
            f"未收到 KeyboardInterrupt：{out}\n{err}",
        )
        self.assertIn("SAVE_REJECTED", out, "取消后 save() 未被拒绝")
        self.assertIn("RESET_RETRAIN_OK", out, "取消后 reset() 重训失败")
        self.assertNotIn("SAVED", out, "取消路径竟然完成了训练")
        self.assertFalse(model_exists, "取消后不能留下半截模型文件")
        self.assertTrue(model2_exists, "reset() 后应能产出正常模型")

        prov = [line for line in out.splitlines() if line.startswith("PROVENANCE")]
        self.assertEqual(len(prov), 1, f"缺少 provenance：{out}")
        _, merges_done, reason, elapsed_ok, stats_ok = prov[0].split()
        self.assertGreaterEqual(int(merges_done), 0)
        self.assertEqual(reason, "signal")
        self.assertEqual(elapsed_ok, "True")
        self.assertEqual(stats_ok, "True")

    def test_custom_signal_handler_propagates(self):
        rc, out, err, model_exists, _ = self._run_child(CHILD_CUSTOM_HANDLER)
        self.assertNotEqual(rc, 0, "自定义 handler 抛出的异常应导致非零退出")
        self.assertIn("MyStop", err, f"自定义异常没有原样传播：{err}")
        self.assertNotIn("SAVED", out, "自定义 handler 抛异常后训练不应继续")
        self.assertFalse(model_exists)


if __name__ == "__main__":
    unittest.main()

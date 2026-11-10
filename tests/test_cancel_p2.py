"""R6-1 P2：CancellationToken / timeout / interruptible / 非主线程告警。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path

from bbpe_tokenizer import (
    BBPETrainer,
    CancellationToken,
    TextFileReader,
    TrainerConfig,
    TrainingCancelled,
    interruptible,
)

LINE = "中文语言文字处理与字节级 BPE 分词器训练 abcabc 中文中文 0123456789 "

CHILD_INTERRUPTIBLE = r'''
import os
import signal
import sys
import threading
import time

import bbpe_tokenizer as b

corpus, model = sys.argv[1:3]
t = b.BBPETrainer(b.TrainerConfig(6000))


def fire():
    # 由子进程自己定时发信号：release 构建的训练比 debug 快数倍，父进程
    # 固定 sleep 后再 kill 会落在训练结束之后。50ms 远小于任何构建下的
    # 训练时长，且 os.kill 与外部 kill 走同一条信号递送路径。
    time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGINT)


with b.interruptible() as tok:
    def run():
        try:
            t.train(b.TextFileReader(corpus), cancel_token=tok)
        except b.TrainingCancelled as e:
            print("BG_CANCELLED", e.reason, e.merges_done, flush=True)

    th = threading.Thread(target=run)
    th.start()
    threading.Thread(target=fire, daemon=True).start()
    print("READY", flush=True)
    try:
        th.join()
    except KeyboardInterrupt:
        print("MAIN_KBI", flush=True)
    # 不依赖第二次 join（CPython 3.12 上 join 被中断后重入不可靠），用轮询
    deadline = time.time() + 10
    while th.is_alive() and time.time() < deadline:
        time.sleep(0.01)
    try:
        t.save(model)
    except RuntimeError:
        print("SAVE_REJECTED", flush=True)
print("DONE", flush=True)
'''


class TestP2TokenAndTimeout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = Path(self.tmp.name) / "corpus.txt"
        self.corpus.write_text((LINE + "\n") * 80000, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _reader(self):
        return TextFileReader(str(self.corpus))

    def test_token_cancel_from_other_thread(self):
        trainer = BBPETrainer(TrainerConfig(4000))
        token = CancellationToken()

        def cancel_soon():
            time.sleep(0.15)
            token.cancel()

        th = threading.Thread(target=cancel_soon)
        th.start()
        with self.assertRaises(TrainingCancelled) as ctx:
            trainer.train(self._reader(), cancel_token=token)
        th.join(10)

        err = ctx.exception
        self.assertEqual(err.reason, "user")
        self.assertGreaterEqual(err.merges_done, 0)
        self.assertGreater(err.elapsed_secs, 0)
        self.assertIsInstance(err.corpus_stats, dict)
        self.assertTrue(token.cancelled)
        with self.assertRaises(RuntimeError):
            trainer.save(str(Path(self.tmp.name) / "no.json"))

        trainer.wait_cleanup()
        trainer.reset()
        trainer.train(TextFileReader(str(self.corpus)))
        self.assertTrue(trainer.is_trained())

    def test_timeout_triggers(self):
        trainer = BBPETrainer(TrainerConfig(4000))
        # 10ms：远小于任何构建下的训练时长（release 也比 debug 快数倍，
        # 用 0.2s 会在 release 下等到训练自然结束而不触发超时）。
        with self.assertRaises(TrainingCancelled) as ctx:
            trainer.train(self._reader(), timeout=0.01)
        self.assertEqual(ctx.exception.reason, "timeout")
        trainer.wait_cleanup()

    def test_timeout_param_validation(self):
        trainer = BBPETrainer(TrainerConfig(320))
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                trainer.train(self._reader(), timeout=bad)
        # 校验发生在 take() 之前，训练态仍可用
        trainer.train(self._reader())
        self.assertTrue(trainer.is_trained())

    def test_non_main_thread_training_warns(self):
        trainer = BBPETrainer(TrainerConfig(320))
        token = CancellationToken()
        captured: list[warnings.WarningMessage] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            def run():
                trainer.train(self._reader(), cancel_token=token)

            th = threading.Thread(target=run)
            th.start()
            th.join(30)
            captured.extend(w)

        self.assertTrue(
            any("非主线程" in str(x.message) for x in captured),
            f"缺少非主线程告警：{[str(x.message) for x in captured]}",
        )

    def test_interruptible_installs_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGINT)
        with interruptible() as tok:
            self.assertIsInstance(tok, CancellationToken)
            self.assertNotEqual(signal.getsignal(signal.SIGINT), previous)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)

    def test_interruptible_falls_back_on_non_main_thread(self):
        results: list[str] = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            def run():
                with interruptible() as tok:
                    results.append(type(tok).__name__)
                    tok.cancel()
                    results.append(str(tok.cancelled))

            th = threading.Thread(target=run)
            th.start()
            th.join(10)
            self.assertTrue(any("非主线程" in str(x.message) for x in w))
        self.assertEqual(results, ["CancellationToken", "True"])


class TestP2BackgroundSigint(unittest.TestCase):
    """后台线程训练 + 主线程 interruptible()，必须解决 shutdown 卡住。"""

    def test_ctrl_c_stops_background_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / "corpus.txt"
            model = root / "model.json"
            corpus.write_text((LINE + "\n") * 200000, encoding="utf-8")

            env = os.environ.copy()
            # 仅当 python/ 内已有编译扩展时才注入源码目录（见 test_boundary_fixes.py 同款说明）
            pkg = Path(__file__).resolve().parents[1] / "python" / "bbpe_tokenizer"
            if list(pkg.glob("bbpe_rs*.so")) or list(pkg.glob("bbpe_rs*.pyd")):
                env["PYTHONPATH"] = os.pathsep.join(
                    p for p in (str(pkg.parent), env.get("PYTHONPATH", "")) if p
                )
            proc = subprocess.Popen(
                [sys.executable, "-c", CHILD_INTERRUPTIBLE, str(corpus), str(model)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                self.assertEqual(proc.stdout.readline().strip(), "READY")
                t0 = time.time()
                # 信号由子进程自己定时发出（见 CHILD_INTERRUPTIBLE.fire）
                out, err = proc.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                self.fail("interruptible() 未能在 20s 内停止后台训练")
            except BaseException:
                proc.kill()
                proc.communicate()
                raise

            # 用户可观察契约：主线程收到 KeyboardInterrupt、训练被取消
            # （save 拒绝、无模型产出）、进程正常收尾。
            self.assertIn("MAIN_KBI", out, f"主线程未收到 KeyboardInterrupt：{out}\n{err}")
            self.assertIn("SAVE_REJECTED", out)
            self.assertIn("DONE", out, "进程未能正常收尾（shutdown 卡住）")
            self.assertFalse(model.exists(), "取消后不能留下模型文件")
            self.assertLess(time.time() - t0, 20)
            # CPython 3.12 上 worker 线程会在取消时被提前终止，来不及把
            # TrainingCancelled 交回 Python 层（连 finally 都不执行）；
            # 3.13 上正常。这不是用户契约的一部分，有则额外校验。
            if "BG_CANCELLED" in out:
                self.assertIn("BG_CANCELLED", out)


if __name__ == "__main__":
    unittest.main()

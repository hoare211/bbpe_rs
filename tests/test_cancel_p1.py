"""R6-1 P1：GIL 释放、状态槽并发访问、cleanup 回压。

这三项是 P1 相对 P0 的核心增量，且都不需要进程级信号（信号链路由
`test_cancel_sigint.py` 覆盖）。
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from bbpe_tokenizer import BBPETrainer, TextFileReader, TrainerConfig

LINE = "中文语言文字处理与字节级 BPE 分词器训练 abcabc 中文中文 0123456789 "


class TestP1Binding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = Path(self.tmp.name) / "corpus.txt"
        # 真实多行语料；该规模训练约 1~2s，足以覆盖下面的线程时序
        self.corpus.write_text((LINE + "\n") * 60000, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_gil_released_during_training(self):
        trainer = BBPETrainer(TrainerConfig(4000))
        started = threading.Event()
        progressed = threading.Event()

        def worker():
            started.set()
            time.sleep(0.05)
            progressed.set()

        th = threading.Thread(target=worker)
        th.start()
        self.assertTrue(started.wait(5), "worker 线程未能启动")

        trainer.train(TextFileReader(str(self.corpus)))

        # 若 train 持 GIL，worker 在训练期间拿不到 GIL、无法开始 sleep；
        # 训练返回后主线程立即检查，事件必为 unset。
        self.assertTrue(
            progressed.is_set(),
            "训练期间未释放 GIL：worker 线程无法推进（m10 回归）",
        )
        th.join(10)

    def test_concurrent_trainer_access_gives_clear_error(self):
        trainer = BBPETrainer(TrainerConfig(4000))
        results: list[str] = []
        done = threading.Event()
        probe = str(Path(self.tmp.name) / "concurrent.json")

        def worker():
            # 不依赖固定 sleep（release 构建训练比 debug 快数倍，sleep 会让
            # 断言在训练结束后才执行）。轮询到「训练中」那一刻为止：训练开始
            # 前 save() 成功（写一个未训练模型），训练开始后必报明确错误。
            while not done.is_set():
                try:
                    trainer.save(probe)
                except RuntimeError as e:
                    results.append(str(e))
                    return
                except BaseException as e:  # noqa: BLE001 - 测试需记录任何异常类型
                    results.append(f"OTHER:{e}")
                    return
                time.sleep(0.002)

        th = threading.Thread(target=worker)
        th.start()
        try:
            trainer.train(TextFileReader(str(self.corpus)))
        finally:
            done.set()
        th.join(10)

        self.assertEqual(len(results), 1, f"并发 save 未按预期报错：{results}")
        self.assertIn("训练", results[0])
        self.assertNotIn("Already borrowed", results[0])

    def test_wait_cleanup_is_noop_after_normal_run(self):
        trainer = BBPETrainer(TrainerConfig(320))
        trainer.wait_cleanup()
        trainer.train(TextFileReader(str(self.corpus)))
        trainer.wait_cleanup()
        self.assertTrue(trainer.is_trained())

    def test_save_works_after_normal_training(self):
        trainer = BBPETrainer(TrainerConfig(320))
        trainer.train(TextFileReader(str(self.corpus)))
        out = Path(self.tmp.name) / "model.json"
        trainer.save(str(out))
        self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()

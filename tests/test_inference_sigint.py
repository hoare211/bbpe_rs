"""R6-1：长文本推理（encode 系列）也能被 Ctrl+C 中断。

短任务走零开销原路径；只有 ≥1MiB 的输入才启用 worker + 主线程信号泵。
取消语义是「放弃本次结果」——encode/decode 是纯函数、无共享状态、无 I/O。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from bbpe_tokenizer import BBPETrainer, TextFileReader, TrainerConfig

LINE = "中文语言文字处理与字节级 BPE 分词器训练 abcabc 中文中文 0123456789 "

CHILD = r'''
import sys
from bbpe_tokenizer.bbpe_rs import BBPETokenizer

model = sys.argv[1]
tok = BBPETokenizer(model)
text = "中文分词训练样本 abcdef0123456789 语言文字处理" * 600000
print("READY", len(text), flush=True)
try:
    tok.encode_ordinary(text)
except KeyboardInterrupt:
    print("INTERRUPTED", flush=True)
else:
    print("DONE", flush=True)
'''


class TestInferenceSigint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        corpus = root / "corpus.txt"
        cls.model = str(root / "model.json")
        corpus.write_text((LINE + "\n") * 3000, encoding="utf-8")
        t = BBPETrainer(TrainerConfig(300))
        t.train(TextFileReader(str(corpus)))
        t.save(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_long_text_encoding_interruptible(self):
        env = os.environ.copy()
        # 仅当 python/ 内已有编译扩展时才注入源码目录（见 test_boundary_fixes.py 同款说明）
        pkg = Path(__file__).resolve().parents[1] / "python" / "bbpe_tokenizer"
        if list(pkg.glob("bbpe_rs*.so")) or list(pkg.glob("bbpe_rs*.pyd")):
            env["PYTHONPATH"] = os.pathsep.join(
                p for p in (str(pkg.parent), env.get("PYTHONPATH", "")) if p
            )
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD, self.model],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            ready = proc.stdout.readline().strip()
            self.assertTrue(ready.startswith("READY"), f"子进程启动失败：{ready}")
            self.assertGreater(int(ready.split()[1]), 1 << 20, "输入未达到长文本阈值")
            time.sleep(0.5)
            t0 = time.time()
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail("长文本编码未能在 15s 内响应 SIGINT")
        except BaseException:
            proc.kill()
            proc.communicate()
            raise

        self.assertIn("INTERRUPTED", out, f"未收到 KeyboardInterrupt：{out}\n{err}")
        self.assertNotIn("DONE", out, "编码不应完成（取消后结果必须被放弃）")
        self.assertLess(time.time() - t0, 15)

    def test_short_text_encoding_unaffected(self):
        """回归：短路径（<1MiB）仍走原 py.detach，结果必须与长路径一致。"""
        from bbpe_tokenizer.bbpe_rs import BBPETokenizer

        tok = BBPETokenizer(self.model)
        text = LINE * 100
        ids = tok.encode_ordinary(text)
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)
        self.assertEqual(tok.decode(ids), text)


if __name__ == "__main__":
    unittest.main()

"""R6-1 边界收尾：采样阶段可中断、held-out/保留采样、路径别名防护。

对应三项修复：
① 采样预读移进 worker + `sample_lines_cancel` 检查点（Ctrl+C 覆盖全流程）；
② 暴露 heldout_rate / heldout_seed / heldout_out / sample_out；
③ 可用内存读取加 MemTotal/cgroup 回退，读不到时显式告警（Rust 侧单测覆盖）。
"""

from __future__ import annotations

import glob
import json
import os
import signal
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

LINE = "中文语言文字处理与字节级 BPE 训练 abcabc 中文中文 0123456789 "
SAMPLE_GLOB = "/tmp/bbpe_py_sample_*"

CHILD_SIGINT_DURING_SAMPLING = r'''
import sys
from bbpe_tokenizer import BBPETrainer, TrainerConfig, TextFileReader

corpus = sys.argv[1]
t = BBPETrainer(TrainerConfig(400))
print("READY", flush=True)
try:
    t.train(TextFileReader(corpus), sample_rate=0.5)
except KeyboardInterrupt:
    print("INTERRUPTED", flush=True)
else:
    print("DONE", flush=True)
'''


def sample_files() -> set[str]:
    return set(glob.glob(SAMPLE_GLOB))


class TestSamplingInterruptible(unittest.TestCase):
    def test_real_sigint_interrupts_sampling_prefetch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / "corpus.txt"
            # ~20MB：采样预读有可观测的持续时间，但不会拖慢测试
            with open(corpus, "w", encoding="utf-8") as f:
                for i in range(280_000):
                    f.write(f"{LINE} 序号{i}\n")

            before = sample_files()
            env = os.environ.copy()
            # 只有 `maturin develop` 之后 python/ 里才有编译好的扩展；干净检出
            # 时只继承环境（包由已安装的 wheel 提供），否则子进程 import 失败。
            pkg = Path(__file__).resolve().parents[1] / "python" / "bbpe_tokenizer"
            if list(pkg.glob("bbpe_rs*.so")) or list(pkg.glob("bbpe_rs*.pyd")):
                env["PYTHONPATH"] = os.pathsep.join(
                    p for p in (str(pkg.parent), env.get("PYTHONPATH", "")) if p
                )
            proc = subprocess.Popen(
                [sys.executable, "-c", CHILD_SIGINT_DURING_SAMPLING, str(corpus)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                self.assertEqual(proc.stdout.readline().strip(), "READY")
                # 等采样临时文件出现 —— 此刻采样循环刚开始，信号必定落在预读阶段
                deadline = time.time() + 20
                while time.time() < deadline and not (sample_files() - before):
                    time.sleep(0.005)
                self.assertNotEqual(
                    sample_files(), before, "采样临时文件未出现，无法定位预读阶段"
                )
                t0 = time.time()
                proc.send_signal(signal.SIGINT)
                out, err = proc.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                self.fail("采样阶段未能在 20s 内响应 SIGINT")
            except BaseException:
                proc.kill()
                proc.communicate()
                raise

            self.assertIn("INTERRUPTED", out, f"未收到 KeyboardInterrupt：{out}\n{err}")
            self.assertNotIn("DONE", out, "取消后训练不应完成")
            self.assertLess(time.time() - t0, 20)
            self.assertTrue(corpus.exists(), "原始语料不得改动")
            # worker 结束时守卫自动清理；给一点宽限但不允许残留
            deadline = time.time() + 2
            while time.time() < deadline and sample_files() != before:
                time.sleep(0.02)
            self.assertEqual(sample_files(), before, "取消后临时样本文件残留")


class TestHeldoutAndSampleOut(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus.txt"
        self.corpus.write_text(
            "".join(f"{LINE} 序号{i}\n" for i in range(12000)), encoding="utf-8"
        )
        self.before = sample_files()

    def tearDown(self):
        self.assertEqual(sample_files(), self.before, "出现未清理的临时样本文件")
        self.tmp.cleanup()

    def _trainer(self):
        return BBPETrainer(TrainerConfig(400))

    def test_heldout_and_kept_sample_outputs(self):
        sample_out = self.root / "sample.jsonl"
        held_out = self.root / "held.jsonl"
        model = self.root / "model.json"
        t = self._trainer()
        t.train(
            TextFileReader(str(self.corpus)),
            sample_rate=0.6,
            sample_seed=5,
            heldout_rate=0.1,
            heldout_seed=9,
            heldout_out=str(held_out),
            sample_out=str(sample_out),
        )
        t.save(str(model))

        data = json.loads(model.read_text(encoding="utf-8"))
        meta = data["meta"]["sampling"]
        self.assertEqual(meta["rate"], 0.6)
        self.assertEqual(meta["seed"], 5)
        self.assertEqual(meta["heldout_rate"], 0.1)
        self.assertEqual(meta["heldout_seed"], 9)

        self.assertTrue(sample_out.exists(), "sample_out 应被保留")
        self.assertTrue(held_out.exists(), "heldout_out 应被保留")
        train_lines = set(sample_out.read_text(encoding="utf-8").splitlines())
        held_lines = set(held_out.read_text(encoding="utf-8").splitlines())
        original = set(self.corpus.read_text(encoding="utf-8").splitlines())
        self.assertTrue(train_lines.isdisjoint(held_lines), "训练/held-out 必须互斥")
        self.assertTrue(train_lines | held_lines <= original)
        self.assertLess(len(train_lines), len(original))

    def test_cancel_leaves_target_untouched_no_part_file(self):
        sample_out = self.root / "sample.jsonl"
        held_out = self.root / "held.jsonl"
        sample_out.write_text("原有采样内容", encoding="utf-8")
        held_out.write_text("原有 held-out 内容", encoding="utf-8")

        tok = CancellationToken()
        tok.cancel()
        t = self._trainer()
        with self.assertRaises(TrainingCancelled):
            t.train(
                TextFileReader(str(self.corpus)),
                cancel_token=tok,
                sample_rate=0.5,
                heldout_rate=0.1,
                heldout_out=str(held_out),
                sample_out=str(sample_out),
            )
        t.wait_cleanup()

        # 原子落盘：取消时目标文件一个字节都不能变
        self.assertEqual(sample_out.read_text(encoding="utf-8"), "原有采样内容")
        self.assertEqual(held_out.read_text(encoding="utf-8"), "原有 held-out 内容")
        self.assertEqual(list(self.root.glob(".*.part.*")), [], "残留 .part 暂存文件")

    def test_success_commits_without_part_file(self):
        sample_out = self.root / "sample.jsonl"
        held_out = self.root / "held.jsonl"
        t = self._trainer()
        t.train(
            TextFileReader(str(self.corpus)),
            sample_rate=0.6,
            sample_seed=5,
            heldout_rate=0.1,
            heldout_seed=9,
            heldout_out=str(held_out),
            sample_out=str(sample_out),
        )
        self.assertTrue(sample_out.exists())
        self.assertTrue(held_out.exists())
        self.assertEqual(list(self.root.glob(".*.part.*")), [], "残留 .part 暂存文件")

    def test_zero_heldout_rate_creates_no_empty_file(self):
        held_out = self.root / "held.jsonl"
        t = self._trainer()
        t.train(
            TextFileReader(str(self.corpus)),
            sample_rate=0.5,
            heldout_out=str(held_out),
        )
        self.assertFalse(held_out.exists(), "rate=0 时不应留下空 held-out 文件")

    def test_param_and_path_alias_validation(self):
        t = self._trainer()
        r = TextFileReader(str(self.corpus))
        good = self.root / "ok.jsonl"

        cases = [
            ({"heldout_rate": 0.1}, "缺 heldout_out"),
            ({"heldout_rate": 0.1, "heldout_out": str(good)}, "缺 sample_rate"),
            ({"heldout_rate": 1.0}, "heldout_rate 越界"),
            ({"sample_out": str(self.corpus), "sample_rate": 0.5}, "覆盖原始语料"),
            (
                {
                    "sample_rate": 0.5,
                    "sample_out": str(good),
                    "heldout_rate": 0.1,
                    "heldout_out": str(good),
                },
                "两输出同路径",
            ),
        ]
        for kw, label in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    t.train(r, **kw)
                t.reset()

        # 校验失败不应留下任何采样文件
        self.assertFalse(good.exists())
        self.assertEqual(sample_files(), self.before)


if __name__ == "__main__":
    unittest.main()

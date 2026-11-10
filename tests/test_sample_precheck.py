"""R6-1 P3：Python 侧采样训练与 OOM 内存预检。

复用 CLI 的 `sampling::sample_lines` 与 `cli::util::exceeds_memory_budget`，
重点验收绑定层编排：临时文件生命周期、`meta.sampling` 据实写入、
预检失败绝不删用户输入、取消路径同样不遗留临时文件。
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
import threading
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

LINE = "中文语言文字处理与字节级 BPE 分词器训练 abcabc 中文中文 0123456789 "
SAMPLE_GLOB = "/tmp/bbpe_py_sample_*"


def sample_files() -> set[str]:
    return set(glob.glob(SAMPLE_GLOB))


class _CorpusCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus.txt"
        # 每行唯一，保证采样后词表仍有足够内容
        self.corpus.write_text(
            "".join(f"{LINE} 序号{i}\n" for i in range(12000)), encoding="utf-8"
        )
        self.before = sample_files()

    def tearDown(self):
        self.assertEqual(sample_files(), self.before, "临时采样文件未清理")
        self.tmp.cleanup()

    def _trainer(self, vocab=400):
        return BBPETrainer(TrainerConfig(vocab))

    def _train_and_save(self, name="model.json", **kw):
        t = self._trainer()
        out = self.root / name
        t.train(TextFileReader(str(self.corpus)), **kw)
        t.save(str(out))
        return json.loads(out.read_text(encoding="utf-8"))


class TestSampling(_CorpusCase):
    def test_sampling_params_written_to_meta_and_temp_cleaned(self):
        data = self._train_and_save(sample_rate=0.5, sample_seed=7)
        sampling = data["meta"]["sampling"]
        self.assertEqual(sampling["rate"], 0.5)
        self.assertEqual(sampling["seed"], 7)
        self.assertEqual(sampling["heldout_rate"], 0.0)
        self.assertGreater(len(data["merges"]), 0)
        # 训练读的是采样产物，字节数必须小于原始语料
        self.assertLess(
            data["meta"]["corpus"]["total_bytes"], self.corpus.stat().st_size
        )

    def test_same_seed_sampling_is_deterministic(self):
        a = self._train_and_save("a.json", sample_rate=0.5, sample_seed=17)
        b = self._train_and_save("b.json", sample_rate=0.5, sample_seed=17)
        self.assertEqual(a["vocab"], b["vocab"])
        self.assertEqual(a["merges"], b["merges"])

    def test_full_training_writes_no_sampling_block(self):
        data = self._train_and_save()
        self.assertNotIn("sampling", data["meta"])

    def test_max_bytes_effective(self):
        cap = 200_000
        data = self._train_and_save(sample_rate=1.0, max_bytes=cap)
        # max_bytes 按累计读入字节在行边界停，产物不应显著超过上限
        self.assertLessEqual(data["meta"]["corpus"]["total_bytes"], cap + len(LINE) * 4)

    def test_sampling_param_validation(self):
        t = self._trainer()
        for bad in (0.0, 1.5, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                t.train(TextFileReader(str(self.corpus)), sample_rate=bad)
            t.reset()

    def test_temp_sample_cleaned_after_cancel(self):
        # 训练前置位，避免小语料下训练早于定时器完成导致的时序抖动；
        # 覆盖的是「采样已发生 → 训练被取消 → guard 仍被清理」这条路径。
        tok = CancellationToken()
        tok.cancel()
        trainer = self._trainer()
        with self.assertRaises(TrainingCancelled):
            trainer.train(
                TextFileReader(str(self.corpus)),
                cancel_token=tok,
                sample_rate=0.6,
                sample_seed=3,
            )
        # 回压后 worker 结果里的 TempFileGuard 才会 drop
        trainer.wait_cleanup()
        self.assertEqual(sample_files(), self.before)

    def test_same_seed_matches_cli(self):
        bin_path = Path(__file__).resolve().parents[1] / "target" / "debug" / "bbpe"
        if not bin_path.exists():
            self.skipTest("需先 cargo build --bin bbpe（cargo test 会自动构建）")
        py_model = self.root / "py.json"
        cli_model = self.root / "cli.json"
        t = self._trainer()
        t.train(
            TextFileReader(str(self.corpus)), sample_rate=0.5, sample_seed=23
        )
        t.save(str(py_model))
        subprocess.run(
            [
                str(bin_path),
                "train",
                "--input",
                str(self.corpus),
                "--format",
                "text",
                "--vocab",
                "400",
                "--output",
                str(cli_model),
                "--sample-rate",
                "0.5",
                "--sample-seed",
                "23",
            ],
            check=True,
            capture_output=True,
        )
        a = json.loads(py_model.read_text(encoding="utf-8"))
        b = json.loads(cli_model.read_text(encoding="utf-8"))
        self.assertEqual(a["merges"], b["merges"])
        self.assertEqual(a["vocab"], b["vocab"])


class TestOomPrecheck(unittest.TestCase):
    def test_precheck_failure_keeps_corpus_and_no_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / "sparse.txt"
            # 稀疏文件：metadata 声明 32GB，但不实际分配块。
            # 9.2 × 32GB 远超可用内存，必然触发预检。
            with open(corpus, "wb") as f:
                f.truncate(32 * 1024**3)
            model = root / "model.json"
            before = sample_files()

            t = BBPETrainer(TrainerConfig(400))
            with self.assertRaises(RuntimeError) as ctx:
                t.train(TextFileReader(str(corpus)), oom_precheck=True)
            self.assertIn("装不进内存", str(ctx.exception))
            self.assertIn("sample_rate", str(ctx.exception))
            self.assertTrue(corpus.exists(), "R4-1 铁律：预检失败不得删原始语料")
            self.assertEqual(corpus.stat().st_size, 32 * 1024**3)
            self.assertFalse(model.exists())
            self.assertEqual(sample_files(), before)
            os.remove(corpus)

    def test_precheck_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / "corpus.txt"
            corpus.write_text((LINE + "\n") * 3000, encoding="utf-8")
            model = root / "model.json"
            t = BBPETrainer(TrainerConfig(320))
            t.train(TextFileReader(str(corpus)), oom_precheck=False)
            t.save(str(model))
            self.assertTrue(model.exists())


if __name__ == "__main__":
    unittest.main()

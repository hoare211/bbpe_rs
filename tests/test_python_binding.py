# tests/test_python_binding.py
"""Rust 绑定的端到端测试（需先 `maturin develop`）。

本文件此前使用的是**早已失效的旧 API**（`BBPETrainer()` 无参构造、
`train(path, vocab_size=..., num_shards=...)`），与当前绑定签名不符，等于
处于"必然 TypeError"状态、实际没有测试。2026-10-07 按当前 API 重写，
并补入两条新回归：

- A1：`BBPETrainer.train` 二次调用必须报错（Python 侧是复现 A1 最自然的路径）
- A3：`encode` 与 `encode_parallel` 对同一 `allowed_special` 必须给出相同 ids
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest

# 注意导入路径：maturin 把 Rust 扩展放在 `bbpe_tokenizer.bbpe_rs`（嵌套），
# 不是顶层的 `bbpe_rs`。原测试写的是 `import bbpe_rs`，必然 ImportError。
from bbpe_tokenizer import bbpe_rs

VOCAB = 256 + 20


class TestRustBBPEBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bbpe_rs.init_rust_logger()
        cls.tmp = tempfile.TemporaryDirectory(prefix="bbpe_py_test_")
        cls.corpus_path = os.path.join(cls.tmp.name, "corpus.txt")
        cls.corpus2_path = os.path.join(cls.tmp.name, "corpus2.txt")
        cls.model_path = os.path.join(cls.tmp.name, "model.json")

        with open(cls.corpus_path, "w", encoding="utf-8") as f:
            f.write("Hello 世界! 这是一个基于 Rust 的分词器测试。\n" * 500)
        # 与上面**完全不同**的语料 —— 用于验证二次训练被正确拒绝（A1）
        with open(cls.corpus2_path, "w", encoding="utf-8") as f:
            f.write("completely different words foo bar baz qux quux\n" * 500)

        cfg = bbpe_rs.TrainerConfig(VOCAB)
        trainer = bbpe_rs.BBPETrainer(cfg)
        trainer.train(bbpe_rs.TextFileReader(cls.corpus_path))
        trainer.save(cls.model_path)

        cls.tokenizer = bbpe_rs.BBPETokenizer(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---------- 基础 ----------

    def test_round_trip(self):
        text = "Hello 世界! 这是一个基于 Rust 的分词器测试。"
        ids = self.tokenizer.encode(text).ids
        self.assertEqual(text, self.tokenizer.decode(ids))

    def test_empty_string(self):
        ids = self.tokenizer.encode("").ids
        self.assertEqual(ids, [])
        self.assertEqual(self.tokenizer.decode(ids), "")

    def test_decode_invalid_id(self):
        with self.assertRaises(ValueError):
            self.tokenizer.decode([99999])

    def test_train_invalid_vocab_size(self):
        # vocab 必须 > 256（还要装得下 special tokens）
        with self.assertRaises(RuntimeError):
            bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(100))

    def test_offsets_are_monotonic_and_non_overlapping(self):
        text = "Hello 世界! 这是一个基于 Rust 的分词器测试。"
        enc = self.tokenizer.encode(text)
        self.assertEqual(len(enc.ids), len(enc.offsets))
        prev_end = 0
        for (start, end) in enc.offsets:
            self.assertLessEqual(start, end)
            self.assertGreaterEqual(start, prev_end, "偏移必须单调不重叠")
            prev_end = end
        self.assertEqual(self.tokenizer.decode(enc.ids), text)

    def test_encode_ordinary_matches_encode_ids(self):
        text = "Hello 世界! 这是一个基于 Rust 的分词器测试。"
        self.assertEqual(self.tokenizer.encode_ordinary(text), self.tokenizer.encode(text).ids)

    # ---------- A1：train 非幂等 ----------

    def test_a1_second_train_must_error(self):
        """A1：同一 trainer 二次训练必须报错，而不是静默产出违反 V3 的模型。

        这是 A1 最自然的复现路径 —— Notebook 里换个语料重训是常见用法。
        """
        trainer = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(VOCAB))
        trainer.train(bbpe_rs.TextFileReader(self.corpus_path))
        with self.assertRaises(RuntimeError) as cm:
            trainer.train(bbpe_rs.TextFileReader(self.corpus2_path))
        # 报错必须可操作（指明 reset 或新建）
        self.assertIn("reset", str(cm.exception).lower())

    def test_a1_new_trainer_trains_independently(self):
        """对照：新建 trainer 不受影响，仍可正常训练。"""
        trainer = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(VOCAB))
        trainer.train(bbpe_rs.TextFileReader(self.corpus2_path))
        out = os.path.join(self.tmp.name, "model2.json")
        trainer.save(out)
        self.assertTrue(os.path.exists(out))

    # ---------- B4：reset() / is_trained() 必须是真实存在的出口 ----------

    def test_b4_reset_retrain_matches_new_trainer(self):
        """B4：A1 的报错让用户"调用 reset()"，该出口在 Python 侧必须真的存在。

        回归背景：Rust 侧已有 `reset()` / `is_trained()`，但 `PyBBPETrainer`
        此前只暴露 new / train / save ⟹ Python 用户（A1 最自然的触发路径：
        Notebook 里换语料重训）会收到一条指向不存在出口的错误。
        同时钉住不变量：reset 后重训 == 全新 trainer 训同一语料。
        """
        t = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(VOCAB))
        self.assertFalse(t.is_trained())
        t.train(bbpe_rs.TextFileReader(self.corpus_path))
        self.assertTrue(t.is_trained())

        # 换语料重训：先 reset 再训，不应报错
        t.reset()
        self.assertFalse(t.is_trained())
        t.train(bbpe_rs.TextFileReader(self.corpus2_path))
        self.assertTrue(t.is_trained())

        out_reset = os.path.join(self.tmp.name, "model_reset.json")
        t.save(out_reset)

        fresh = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(VOCAB))
        fresh.train(bbpe_rs.TextFileReader(self.corpus2_path))
        out_fresh = os.path.join(self.tmp.name, "model_fresh.json")
        fresh.save(out_fresh)

        with open(out_reset, encoding="utf-8") as f:
            a = json.load(f)
        with open(out_fresh, encoding="utf-8") as f:
            b = json.load(f)
        self.assertEqual(
            a["merges"], b["merges"],
            "reset 后重训必须与全新 trainer 逐条等价（否则 reset 不彻底）",
        )
        self.assertEqual(a["vocab"], b["vocab"], "vocab 也必须一致")

    # ---------- A3：串行 / 并行必须一致 ----------

    def test_a3_serial_and_parallel_special_agree(self):
        """A3：`encode_parallel` 曾漏改 m7，与 `encode` 行为分叉。

        回归背景：`allowed_special=["<pad>", "all"]` 时，`encode` 放行全部
        special，而 `encode_parallel` 一个都不放行 —— 同一输入两个 API 给出
        不同 ids，防注入语义在并行路径被静默降级。
        """
        text = "a" * 6000 + "Hello 世界"
        for allowed in (["all"], ["all", "all"], ["<pad>", "all"], ["<|im_start|>"], None):
            a = self.tokenizer.encode(text, allowed_special=allowed)
            b = self.tokenizer.encode_parallel(text, allowed_special=allowed, num_threads=2)
            self.assertEqual(
                a.ids, b.ids, f"串行与并行对 allowed_special={allowed} 给出不同 ids"
            )
            self.assertEqual(a.offsets, b.offsets, f"offsets 不一致：allowed={allowed}")

    def test_a3_long_text_serial_parallel_agree(self):
        text = ("Hello 世界! 这是一个基于 Rust 的分词器测试。\n" * 500) + "x" * 6000
        a = self.tokenizer.encode(text)
        b = self.tokenizer.encode_parallel(text, num_threads=2)
        self.assertEqual(a.ids, b.ids)
        self.assertEqual(a.offsets, b.offsets)
        self.assertEqual(self.tokenizer.decode(b.ids), text)

    def test_encode_ordinary_parallel_matches_serial(self):
        text = "Hello 世界! " * 1000
        self.assertEqual(
            self.tokenizer.encode_ordinary_parallel(text, num_threads=2),
            self.tokenizer.encode_ordinary(text),
        )

    # ---------- UTF-8 状态机 ----------

    def test_utf8_props_of_single_han_char(self):
        # "中" = 0xE4 0xB8 0xAD ⟹ 首字节挂 2 个延续字节
        enc = self.tokenizer.encode_ordinary("中")
        # 逐 token 累加：整串编码后 pending 应归零
        pending = 0
        for tid in enc:
            provides, hangs = self.tokenizer.get_utf8_token_props(tid)
            if provides == pending:
                pending = hangs
            elif provides < pending:
                pending = max(0, pending - provides)
            else:
                pending = hangs
        self.assertEqual(pending, 0, "整串编码后不应残留悬空 UTF-8 字节")

    def test_valid_continuations_non_empty(self):
        for pending in range(4):
            self.assertIsInstance(self.tokenizer.get_valid_continuations(pending), list)

    # ---------- R4-16 / R5-5：推理入口必须释放 GIL ----------

    def _assert_gil_released(self, encode_fn):
        """起一个 worker 线程：编码期间它必须能推进（sleep 后 set 事件）。

        若入口持 GIL，worker 在整个编码期间拿不到 GIL、无法开始 sleep；
        编码返回后主线程立即检查事件（不让出 GIL），此时事件必为 unset。
        """
        started = threading.Event()
        progressed = threading.Event()

        def worker():
            started.wait(10.0)
            time.sleep(0.05)
            progressed.set()

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        started.set()
        # ~22 MB 文本：release 下编码耗时远大于 worker 的 0.05s sleep
        text = "Hello 世界! 这是一个基于 Rust 的分词器测试。" * 400_000
        ids = encode_fn(text)
        self.assertTrue(ids, "编码产出不应为空")
        self.assertTrue(
            progressed.is_set(),
            "编码期间未释放 GIL：worker 线程无法推进（R4-16 回归）",
        )
        th.join(10.0)

    def test_r4_16_serial_encode_releases_gil(self):
        self._assert_gil_released(self.tokenizer.encode_ordinary)

    def test_r4_16_parallel_encode_releases_gil(self):
        self._assert_gil_released(
            lambda t: self.tokenizer.encode_ordinary_parallel(t, num_threads=2)
        )

    # ---------- R4-4 / R5-5：ordinary 入口预分词失败必须抛异常 ----------

    def test_r4_4_ordinary_pretokenize_failure_raises(self):
        """自定义 pattern 未覆盖全部文本时，Python 绑定必须抛 RuntimeError，
        而不是返回空/不完整结果且仅 stderr 告警（机器调用方无法感知）。"""
        corpus = os.path.join(self.tmp.name, "lower.txt")
        with open(corpus, "w", encoding="utf-8") as f:
            f.write("abcdefghijklmnopqrstuvwxyz\n" * 60)
        cfg = bbpe_rs.TrainerConfig(300).with_pattern(
            bbpe_rs.PatternFlavor.custom("[a-z]+")
        )
        trainer = bbpe_rs.BBPETrainer(cfg)
        trainer.train(bbpe_rs.TextFileReader(corpus))
        model = os.path.join(self.tmp.name, "lower_model.json")
        trainer.save(model)
        tok = bbpe_rs.BBPETokenizer(model)

        with self.assertRaises(RuntimeError):
            tok.encode_ordinary("abc中文")
        with self.assertRaises(RuntimeError):
            tok.encode_ordinary_parallel("abc中文", num_threads=2)
        # 对照：合法文本两条路径一致且与 encode 同口径
        self.assertEqual(tok.encode_ordinary("abc"), tok.encode("abc").ids)
        self.assertEqual(
            tok.encode_ordinary_parallel("abc", num_threads=2),
            tok.encode_ordinary("abc"),
        )

    # ---------- R4-7 / R5-5：dynamic reader 的 IO 错误可从 Python 直达 ----------

    @unittest.skipUnless(sys.platform == "linux", "EISDIR 故障注入仅 Linux 确定可用")
    def test_r4_7_dynamic_reader_io_error_fails_training(self):
        """Linux 上目录可 open 但 read 报 EISDIR ⟹ 确定性 IO 故障。
        评审点名的直达路径：JsonlFileReader.dynamic ⟹ 训练必须显式失败。"""
        trainer = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(300))
        reader = bbpe_rs.JsonlFileReader.dynamic(
            self.tmp.name, {"text": ("/text", "str")}
        )
        with self.assertRaises(RuntimeError) as cm:
            trainer.train(reader)
        self.assertIn("IO", str(cm.exception))
        self.assertFalse(trainer.is_trained())

    # ---------- 2026-11-08：Rust ↔ Python 接口面核对（防"Rust 有、Python 没写"） ----------

    def test_api_surface_snapshot(self):
        """把 Rust 侧已实现、Python **必须**可达的方法钉成快照。

        背景：2026-11-08 逐项核对发现 `RuntimeConfig.max_chunk_bytes`、
        `PreTokenizer`（pattern 预览）、`encode_ordinary_batch`、模型体检/裁剪
        等能力在绑定层完全缺失 —— 用户视角就是"Rust 明明实现了，Python 却没有"。
        本测试对 `dir()` 做断言：新增 Rust 能力若忘了写绑定，这里直接红。
        """
        expect = {
            bbpe_rs.TrainerConfig: {
                "with_pattern",
                "with_min_frequency",
                "with_special_tokens",
                "vocab_size",
                "min_frequency",
                "special_tokens",
                "pattern",
            },
            bbpe_rs.RuntimeConfig: {
                "with_batch_size",
                "with_num_shards",
                "with_parallel_index",
                "with_max_chunk_bytes",
                "batch_size",
                "num_shards",
                "parallel_index",
                "max_chunk_bytes",
            },
            bbpe_rs.BBPETokenizer: {
                "encode",
                "encode_ordinary",
                "encode_ordinary_batch",
                "encode_parallel",
                "encode_ordinary_parallel",
                "decode",
                "get_utf8_token_props",
                "get_valid_continuations",
                "is_special_id",
                "is_safe_to_truncate_after",
                "cut_to_merges",
                "rewrite_pattern",
                "compare",
                "validate",
                "meta",
                "vocab_size",
                "pattern",
                "format",
                "format_version",
                "save",
            },
            bbpe_rs.BBPETrainer: {
                "train",
                "set_pre_tokenizer",
                "save",
                "reset",
                "is_trained",
                "wait_cleanup",
            },
            bbpe_rs.PreTokenizer: {
                "pre_tokenize",
                "pre_tokenize_checked",
                "pattern",
                "max_chunk_bytes",
            },
            bbpe_rs.TextFileReader: {"iter_lines", "filepath", "kind", "file_size"},
            bbpe_rs.JsonlFileReader: {
                "static_text",
                "static_conversations",
                "dynamic",
                "iter_lines",
                "filepath",
                "kind",
                "file_size",
            },
        }
        for cls, names in expect.items():
            got = {n for n in dir(cls) if not n.startswith("_")}
            missing = sorted(names - got)
            self.assertFalse(missing, f"{cls.__name__} 缺少绑定：{missing}")
        self.assertTrue(hasattr(bbpe_rs, "sample_lines"), "缺少 sample_lines 函数")
        self.assertTrue(hasattr(bbpe_rs, "compare_models"), "缺少 compare_models 函数")

    # ---------- 合并序列对照（等价 bbpe compare） ----------

    def test_compare_self_and_cut_are_identical(self):
        """字节空间对照的两条基本不变量：自比 100%、cut 是纯截断。"""
        report = bbpe_rs.compare_models(self.model_path, self.model_path, at=[5, 10])
        self.assertIsNone(report["first_divergence"])
        self.assertEqual(report["merges_a"], report["merges_b"])
        self.assertEqual(report["positional"]["matched"], report["compared"])
        self.assertEqual(report["positional"]["ratio"], 1.0)
        self.assertEqual(report["a_path"], self.model_path)
        for seg in report["at"]:
            self.assertEqual(seg["positional_ratio"], 100.0)
            self.assertEqual(seg["set_overlap_ratio"], 100.0)
            # id 空间在同模型自比时同样 100%（对照项，不参与结论）
            self.assertEqual(seg["id_space_ratio"], 100.0)

        cut = self.tokenizer.cut_to_merges(5)
        r2 = self.tokenizer.compare(cut, at=[5])
        self.assertIsNone(r2["first_divergence"], "cut 必须只是截断，不得改变前 k 次合并")
        self.assertLessEqual(r2["compared"], 5)

    def test_compare_detects_divergence_across_corpora(self):
        trainer = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(VOCAB))
        trainer.train(bbpe_rs.TextFileReader(self.corpus2_path))
        other = os.path.join(self.tmp.name, "compare_other.json")
        trainer.save(other)

        report = bbpe_rs.compare_models(self.model_path, other, at=[5])
        self.assertIsNotNone(report["first_divergence"], "不同语料训练必然分叉")
        detail = report["divergence"]
        self.assertEqual(detail["index"], report["first_divergence"])
        self.assertEqual(len(detail["a"]), 2)
        self.assertEqual(len(detail["b"]), 2)
        self.assertLess(report["positional"]["ratio"], 1.0)

    def test_compare_ratio_is_none_when_nothing_to_compare(self):
        """任一侧 merges 为 0 ⟹ ratio 必须是 None（不是 NaN，也不是 0）。"""
        tiny_corpus = os.path.join(self.tmp.name, "tiny.txt")
        with open(tiny_corpus, "w", encoding="utf-8") as f:
            f.write("ab\n")  # pair 频次 1 < min_frequency=2 ⟹ 0 merges
        trainer = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(VOCAB))
        trainer.train(bbpe_rs.TextFileReader(tiny_corpus))
        tiny_model = os.path.join(self.tmp.name, "tiny_model.json")
        trainer.save(tiny_model)

        report = bbpe_rs.compare_models(self.model_path, tiny_model, at=[5])
        self.assertEqual(report["compared"], 0)
        self.assertIsNone(
            report["positional"]["ratio"], "无可比较项时必须为 None —— NaN 无法 JSON 序列化"
        )
        self.assertEqual(report["at"], [], "k 被钳制到 0 的观测点必须跳过")

    def test_plain_text_as_jsonl_fails(self):
        """G4：0 行可用时不得静默产出"256 拖 0 merges"的空模型。

        回归背景：把纯文本用 `JsonlFileReader.static_text` 读入时每行都解析失败，
        此前训练照常返回成功，用户拿到一个什么都编码不了的模型文件。
        """
        plain = os.path.join(self.tmp.name, "plain.txt")
        with open(plain, "w", encoding="utf-8") as f:
            f.write("hello world\n你好 世界\n")

        trainer = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(VOCAB))
        with self.assertRaises(RuntimeError) as cm:
            trainer.train(bbpe_rs.JsonlFileReader.static_text(plain))
        msg = str(cm.exception)
        self.assertIn("语料不可用", msg)
        self.assertIn("TextFileReader", msg, "错误里必须给出 Python 侧的可操作处置")
        self.assertFalse(trainer.is_trained(), "失败后不得留下可保存的训练态")

        # 对照：有重复片段的纯文本用 TextFileReader 可正常训练出 merges
        repeat = os.path.join(self.tmp.name, "repeat.txt")
        with open(repeat, "w", encoding="utf-8") as f:
            f.write("hahaha\nhahaha\nhahaha\n")
        ok = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(VOCAB))
        stats = ok.train(bbpe_rs.TextFileReader(repeat))
        self.assertTrue(ok.is_trained())
        self.assertEqual(stats["skipped_lines"], 0)

    def test_config_read_write_roundtrip(self):
        """配置项此前"只写不读"，设完就取不回来（无法断言实际生效值）。"""
        cfg = bbpe_rs.TrainerConfig(300).with_min_frequency(3)
        self.assertEqual(cfg.vocab_size, 300)
        self.assertEqual(cfg.min_frequency, 3)
        self.assertEqual(cfg.special_tokens, [])
        rt = bbpe_rs.RuntimeConfig().with_max_chunk_bytes(128).with_batch_size(777)
        self.assertEqual(rt.max_chunk_bytes, 128)
        self.assertEqual(rt.batch_size, 777)

    def test_train_returns_corpus_stats(self):
        """`skipped_lines` / `io_errors` 是"语料被改动/截断"的唯一信号，必须可断言。"""
        trainer = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(VOCAB))
        stats = trainer.train(bbpe_rs.TextFileReader(self.corpus_path))
        self.assertEqual(
            set(stats),
            {
                "total_lines",
                "total_bytes",
                "skipped_lines",
                "replacement_chars",
                "io_errors",
            },
        )
        self.assertGreater(stats["total_lines"], 0)
        self.assertEqual(stats["io_errors"], 0)

    def test_pretokenizer_full_coverage_check(self):
        """pattern 未覆盖全部文本必须可**事前**发现（此前只有编码时才静默丢字）。"""
        pt = bbpe_rs.PreTokenizer(r"\p{L}+")
        self.assertEqual([b for b, _ in pt.pre_tokenize("abc")], [b"abc"])
        # 非严格：未覆盖的 "123" 被静默丢弃（与 Rust pre_tokenize 同口径）
        self.assertEqual([b for b, _ in pt.pre_tokenize("abc123")], [b"abc"])
        # 严格：必须抛错，且消息指出缺失内容
        with self.assertRaises(RuntimeError) as cm:
            pt.pre_tokenize_checked("abc123")
        self.assertIn("未覆盖", str(cm.exception))
        # 全覆盖 pattern 在严格路径下与原文逐字节一致
        full = bbpe_rs.PreTokenizer(r"\p{L}+|\s+|[^\s\p{L}]")
        joined = b"".join(b for b, _ in full.pre_tokenize_checked("ab cd!"))
        self.assertEqual(joined, "ab cd!".encode())

    def test_reader_iteration_and_metadata(self):
        reader = bbpe_rs.TextFileReader(self.corpus_path)
        self.assertEqual(reader.kind, "text")
        self.assertGreater(reader.file_size, 0)
        self.assertEqual(reader.filepath, self.corpus_path)
        lines = list(reader.iter_lines())
        self.assertGreater(len(lines), 0)
        self.assertIn("Hello", lines[0])

    def test_model_read_surface_and_edit(self):
        tk = self.tokenizer
        self.assertGreater(tk.vocab_size, 256)
        self.assertEqual(tk.format, "bbpe-rs-model")
        self.assertEqual(tk.format_version, 1)
        self.assertTrue(tk.pattern)
        tk.validate()
        meta = tk.meta()
        self.assertIsInstance(meta, dict)
        self.assertIn("train", meta)

        # cut：按 merges 截断后词表必须变小，且原实例不受影响
        before = tk.vocab_size
        cut = tk.cut_to_merges(5)
        self.assertLessEqual(cut.vocab_size, before)
        self.assertEqual(tk.vocab_size, before)
        cut.validate()

        # rewrite_pattern：返回新实例，原实例 pattern 保持不变
        new_pat = r"\p{L}+|\s+|[^\s\p{L}]"
        re_tk = tk.rewrite_pattern(new_pat)
        self.assertEqual(re_tk.pattern, new_pat)
        self.assertNotEqual(tk.pattern, new_pat)

    def test_batch_encode_and_truncate_guard(self):
        texts = ["Hello 世界", "abc", ""]
        batch = self.tokenizer.encode_ordinary_batch(texts)
        self.assertEqual(len(batch), len(texts))
        for text, ids in zip(texts, batch):
            self.assertEqual(ids, self.tokenizer.encode_ordinary(text))

        ids = self.tokenizer.encode_ordinary("Hello")
        self.assertFalse(self.tokenizer.is_special_id(ids[0]))
        self.assertTrue(self.tokenizer.is_safe_to_truncate_after(ids[-1]))

    def test_sample_lines_function(self):
        out = os.path.join(self.tmp.name, "sampled.txt")
        stats = bbpe_rs.sample_lines(self.corpus_path, out, rate=0.5, seed=7)
        self.assertEqual(
            set(stats),
            {
                "total_lines",
                "total_bytes",
                "train_lines",
                "train_bytes",
                "held_lines",
                "held_bytes",
            },
        )
        self.assertTrue(os.path.exists(out))
        self.assertGreater(stats["train_lines"], 0)
        # R4-22：输入/输出别名必须拒绝（否则边读边写同一文件）
        with self.assertRaises(ValueError):
            bbpe_rs.sample_lines(self.corpus_path, self.corpus_path, rate=0.5)

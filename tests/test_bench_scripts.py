# tests/test_bench_scripts.py
"""第五轮评审 R5-5：bench 脚本类回归（此前完全无自动化覆盖）。

脚本位于中文目录 `bench/200G规模语料/`，用 importlib 按文件路径导入。
纯函数（rebuild_from_offsets / parse_jsonl_texts / build_bench_text）
不依赖 Rust 绑定即可测；check_offsets / check_special 需真实分词器，
绑定缺失时 skip（与脚本自身的导入容错同口径）。
"""
import importlib.util
import json
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "bench" / "200G规模语料"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"bench_{name}", BENCH_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def eval_quality():
    return _load_module("eval_quality")


@pytest.fixture(scope="module")
def bench_encode():
    return _load_module("bench_encode")


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    """真实小模型（check_offsets / check_special 用）。"""
    bbpe_rs = pytest.importorskip("bbpe_tokenizer.bbpe_rs")
    d = tmp_path_factory.mktemp("bench_scripts")
    corpus = d / "corpus.txt"
    corpus.write_text(
        "人工智能是计算机科学的一个分支\n机器学习需要大量高质量语料\n" * 200,
        encoding="utf-8",
    )
    cfg = bbpe_rs.TrainerConfig(320).with_special_tokens(["<|s|>"])
    trainer = bbpe_rs.BBPETrainer(cfg)
    trainer.train(bbpe_rs.TextFileReader(str(corpus)))
    model = d / "model.json"
    trainer.save(str(model))
    return bbpe_rs.BBPETokenizer(str(model)), str(model)


# ---------- R4-13 / R5-5：非 ASCII 字节偏移重建 ----------

def test_rebuild_from_offsets_non_ascii_roundtrip(eval_quality):
    text = "你好，世界！abc😀混合 text"
    tb = text.encode("utf-8")
    # 按字符边界构造字节区间（模拟 Rust offsets 的字节语义）
    offsets = []
    pos = 0
    for ch in text:
        n = len(ch.encode("utf-8"))
        offsets.append((pos, pos + n))
        pos += n
    assert pos == len(tb)
    assert eval_quality.rebuild_from_offsets(text, offsets) == text
    # 整段单区间同理
    assert eval_quality.rebuild_from_offsets(text, [(0, len(tb))]) == text


def test_rebuild_from_offsets_char_index_pitfall(eval_quality):
    # 若按 Python 字符下标切（旧 bug），"你好" 的 (0,2) 会重建出 "你好"；
    # 按字节语义 (0,2) 落在 "你" 的 3 字节内部 ⟹ 整体非法切片返回 ""
    assert eval_quality.rebuild_from_offsets("你好", [(0, 2)]) == ""
    assert eval_quality.rebuild_from_offsets("你好", [(0, 3)]) == "你"
    # 越界切片被 Python 钳位，重建仍等于原文；
    # 逐区间的边界合法性由 check_offsets 的 0<=s<e<=len 检查负责
    assert eval_quality.rebuild_from_offsets("abc", [(0, 99)]) == "abc"


def test_rebuild_from_offsets_merge_token_across_boundary(eval_quality):
    """字节级 BPE 的 merge token 可跨字符边界（如 [E4 B8] 是“你”的前两字节）。
    逐区间 decode 会对这种合法编码误报失败；先拼字节再整体解码才正确。"""
    # "你" = E4 BD A0：拆成 (0,1)+(1,3) 两个区间，单区间均不合法、拼接后合法
    assert eval_quality.rebuild_from_offsets("你", [(0, 1), (1, 3)]) == "你"
    assert eval_quality.rebuild_from_offsets("你好", [(0, 2), (2, 6)]) == "你好"


def test_check_offsets_with_real_tokenizer(eval_quality, tok):
    tokenizer, _ = tok
    assert eval_quality.check_offsets(tokenizer, eval_quality.HARD_CASES) == 0


def test_check_special_semantics_and_failed_flag(eval_quality, tok):
    tokenizer, model_path = tok
    specials = eval_quality.load_special_tokens(model_path)
    assert specials == ["<|s|>"]
    res = eval_quality.check_special(tokenizer, specials)
    assert res["unauthorized_fragmented"] is True
    assert res["authorized_atomic"] is True
    assert res["failed"] is False
    # 无 special 的模型：skipped 且不计失败
    res_empty = eval_quality.check_special(tokenizer, [])
    assert res_empty["failed"] is False


# ---------- R5-4：坏行跳过并计数 ----------

def test_parse_jsonl_texts_counts_bad_lines(bench_encode):
    raw = [
        json.dumps({"text": "你好"}),
        "{bad json",
        json.dumps({"no_text_field": 1}),
        json.dumps({"text": 123}),  # text 非字符串
        "[1, 2, 3]",  # 非 dict
        "   ",
        json.dumps({"text": "world"}),
    ]
    lines, counts = bench_encode.parse_jsonl_texts(raw)
    assert lines == ["你好", "world"]
    assert counts == {"ok": 2, "bad_json": 1, "no_text": 3, "empty": 1}


def test_parse_jsonl_texts_all_bad_no_crash(bench_encode):
    lines, counts = bench_encode.parse_jsonl_texts(["{bad", ""])
    assert lines == []
    assert counts["bad_json"] == 1 and counts["empty"] == 1


def test_build_bench_text_consumes_parsed(bench_encode):
    # R4-25：拼接的是解析后的文本，而非整行 JSON（不含引号/花括号/键名）
    text = bench_encode.build_bench_text(["你好", "world"], min_bytes=20, cap_bytes=200)
    assert "{" not in text and '"text"' not in text
    assert "你好" in text and "world" in text
    assert len(text.encode("utf-8")) >= 20
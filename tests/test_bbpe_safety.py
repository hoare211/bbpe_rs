# tests/test_bbpe_safety.py
"""第五轮评审 R5-5 的安全类盲区回归（pytest 风格）。

覆盖第四轮修复后仍敞开的测试盲区：
- R4-18：SFT 的 BOS/EOS 未注册为 special 时必须抛 ValueError（而非静默取首 ID）
- R4-19：chat template 沙箱逃逸（`{{ ''.__class__ }}`）必须被中和（jinja2 3.1.x
  沙箱将不安全属性访问转为 Undefined，深层爬链显式报错）
- R4-17：FSM 前导字节级承诺在 Python 侧的序列级行为（与 Rust 单测同口径）
- logits 处理器实例化与掩码生效（需 transformers，未装则 skip）
"""
import json

import pytest

from bbpe_tokenizer import bbpe_rs
from bbpe_tokenizer.config import TokenizerGlobalConfig
from bbpe_tokenizer.core import BBTokenizer


@pytest.fixture(scope="module")
def plain_core(tmp_path_factory):
    """无 special 的常规模型（SFT 负例用）。"""
    d = tmp_path_factory.mktemp("safety_plain")
    corpus = d / "corpus.txt"
    corpus.write_text("Hello 世界! 这是一个基于 Rust 的分词器测试。\n" * 300, encoding="utf-8")
    trainer = bbpe_rs.BBPETrainer(bbpe_rs.TrainerConfig(300))
    trainer.train(bbpe_rs.TextFileReader(str(corpus)))
    model = d / "model.json"
    trainer.save(str(model))
    return bbpe_rs.BBPETokenizer(str(model))


@pytest.fixture(scope="module")
def sft_core(tmp_path_factory):
    """注册了 BOS/EOS 的模型（SFT 正例用）。"""
    d = tmp_path_factory.mktemp("safety_sft")
    corpus = d / "corpus.txt"
    corpus.write_text("Hello 世界! 这是一个基于 Rust 的分词器测试。\n" * 300, encoding="utf-8")
    cfg = bbpe_rs.TrainerConfig(300).with_special_tokens(["<|im_start|>", "<|im_end|>"])
    trainer = bbpe_rs.BBPETrainer(cfg)
    trainer.train(bbpe_rs.TextFileReader(str(corpus)))
    model = d / "model.json"
    trainer.save(str(model))
    return bbpe_rs.BBPETokenizer(str(model))


@pytest.fixture(scope="module")
def fsm_core(tmp_path_factory):
    """手工构造的 FSM 探针模型：把关键字节序列放进词表，直接观测属性。

    256=[C0]（非法前导）  257=[C2]（合法 2 字节前导）
    258=[F5]（超范围前导） 259=[80]（延续字节）
    260=[ED A0 80]（surrogate） 261=[C0 80]（overlong）
    """
    d = tmp_path_factory.mktemp("safety_fsm")
    vocab = {str(i): [i] for i in range(256)}
    vocab.update({
        "256": [0xC0],
        "257": [0xC2],
        "258": [0xF5],
        "259": [0x80],
        "260": [0xED, 0xA0, 0x80],
        "261": [0xC0, 0x80],
    })
    model = {
        "format": "bbpe-rs-model",
        "format_version": 1,
        "vocab": vocab,
        "merges": {},
        "pre_tokenizer_pattern": r"[a-z]+",
    }
    path = d / "fsm.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    return bbpe_rs.BBPETokenizer(str(path))


# ---------- R4-18：SFT BOS/EOS 未注册必须显式失败 ----------

def test_r4_18_sft_unregistered_bos_eos_raises(plain_core):
    tok = BBTokenizer(plain_core, TokenizerGlobalConfig())
    # 默认 bos=<|im_start|> / eos=<|im_end|> 均未注册 ⟹ 编码成多个普通 token
    with pytest.raises(ValueError, match="bos_token"):
        tok.encode_dialogue_safely([
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "您好"},
        ])


def test_r4_18_sft_registered_bos_eos_ok(sft_core):
    tok = BBTokenizer(sft_core, TokenizerGlobalConfig())
    out = tok.encode_dialogue_safely([
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好"},
    ])
    ids, labels = out["input_ids"], out["labels"]
    assert len(ids) == len(labels) and len(ids) > 0
    # user 回合整体掩码；assistant 回合的 content 必须参与 loss
    assert -100 in labels
    assert any(l != -100 for l in labels)


# ---------- R4-19：chat template 沙箱逃逸必须被拦截 ----------

def test_r4_19_sandbox_escape_blocked(plain_core):
    import jinja2
    from jinja2.exceptions import SecurityError

    # 对照：普通 Environment 下该探针能拿到 str 类（证明探针有效）
    plain = jinja2.Environment().from_string("{{ ''.__class__ }}").render()
    assert plain == "<class 'str'>"

    tok = BBTokenizer(plain_core, TokenizerGlobalConfig())
    # jinja2 3.1.x 沙箱把不安全属性访问中和为 Undefined（渲染为空），
    # 而非直接抛 SecurityError —— 安全属性是“拿不到 str 类”，不是报错形态
    tok.config.chat_template = "{{ ''.__class__ }}"
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "你好"}], tokenize=False
    )
    assert "<class 'str'>" not in rendered

    # 更深的爬链（__mro__ → __subclasses__）必须显式失败
    tok.config.chat_template = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    with pytest.raises((SecurityError, jinja2.UndefinedError)):
        tok.apply_chat_template(
            [{"role": "user", "content": "你好"}], tokenize=False
        )


def test_r4_19_normal_template_renders(sft_core):
    tok = BBTokenizer(sft_core, TokenizerGlobalConfig())
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "你好"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    assert "user" in rendered and "你好" in rendered
    assert rendered.endswith("assistant\n")


# ---------- R4-17：FSM 前导字节级承诺（Python 侧序列级观测） ----------

def test_r4_17_fsm_leading_byte_props(fsm_core):
    props = fsm_core.get_utf8_token_props
    # 非法前导不开启悬空（C0/C1 overlong 前导、F5–F7 超范围前导）
    assert props(256) == (0, 0)  # [C0]
    assert props(258) == (0, 0)  # [F5]
    # 合法前导开启悬空
    assert props(257) == (0, 1)  # [C2]
    # 延续字节：provides=1
    assert props(259) == (1, 0)  # [80]
    # 序列级边界（与 Rust 单测同口径）：surrogate / overlong 在**计数层面**
    # 被当作完整字符消费 —— 这是前导字节级承诺的既定边界，不是序列级校验器
    assert props(260) == (0, 0)  # [ED A0 80] surrogate
    assert props(261) == (0, 0)  # [C0 80] overlong


def test_r4_17_valid_continuations_match_leading(fsm_core):
    cont0 = fsm_core.get_valid_continuations(0)
    cont1 = fsm_core.get_valid_continuations(1)
    # 悬空 1 字节时：延续字节可补齐；非法前导不悬空故也可选
    assert 259 in cont1  # [80] 补齐 C2 的悬空
    assert 259 not in cont0  # pending=0 时延续字节无处安放
    assert 257 in cont0  # 合法前导可在任意时刻开启新字符
    assert 256 in cont0 and 258 in cont0  # 非法前导不悬空 ⟹ 不阻断生成


# ---------- logits 处理器（需 transformers；环境未装则 skip） ----------

def test_logits_processor_instantiation_and_mask(fsm_core):
    pytest.importorskip("transformers")
    import torch

    from bbpe_tokenizer.logits_processor import UTF8StrictLogitsProcessor

    tok = BBTokenizer(fsm_core, TokenizerGlobalConfig())
    proc = UTF8StrictLogitsProcessor(tok, batch_size=1, device="cpu")

    vocab_size = 262
    scores = torch.zeros(1, vocab_size)
    # 末 token = 257（[C2]，hangs=1）⟹ 下一步只允许延续字节
    input_ids = torch.tensor([[257]], dtype=torch.long)
    out = proc(input_ids, scores)
    assert torch.isfinite(out[0, 259]), "延续字节 [80] 必须被放行"
    assert out[0, 256] == -float("inf"), "非法前导 [C0] 在悬空态必须被掩掉"
    assert out[0, 257] == -float("inf"), "悬空未补齐时不得开启新前导"
    assert proc.violations == 0
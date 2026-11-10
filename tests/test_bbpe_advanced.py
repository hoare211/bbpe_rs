import pytest
import tempfile
import os
import json

from bbpe_tokenizer import (
    BBTokenizer,
    BBPETrainer,
    TrainerConfig,
    JsonlFileReader,
    TextFileReader
)

# ==============================================================================
# Fixtures: 动态准备测试环境，保证每次测试干净独立
# ==============================================================================

@pytest.fixture(scope="module")
def temp_tokenizer_dir():
    """在临时目录中训练一个包含特定特殊字符的 Tokenizer"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. 构造微型语料
        corpus_path = os.path.join(tmp_dir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("Hello 世界! 执行 <tool_call> 测试 </tool> 龘龘龘" * 50)

        # 2. 构造具有"重叠关系"的特殊字符集，测试最长匹配
        special_tokens = ["<tool>", "<tool_call>", "</tool>", "tool_call", "all"]
        vocab_size = 280

        t_config = TrainerConfig(vocab_size=vocab_size).with_special_tokens(special_tokens)
        trainer = BBPETrainer(trainer_config=t_config)
        reader = TextFileReader(corpus_path)

        # 训练并保存基础模型
        trainer.train(reader)
        model_path = os.path.join(tmp_dir, "tokenizer.json")
        trainer.save(model_path)

        # 3. 【核心修复】：模拟业务流水线，将特殊字符手动注入到 added_tokens 字段
        with open(model_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        base_vocab_size = vocab_size - len(special_tokens)
        added_tokens = {}
        for i, token in enumerate(special_tokens):
            added_tokens[token] = base_vocab_size + i

        tokenizer_data["added_tokens"] = added_tokens

        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

        yield tmp_dir

@pytest.fixture(scope="module")
def tokenizer(temp_tokenizer_dir):
    # 初始化封装好的高级类
    return BBTokenizer.from_pretrained(temp_tokenizer_dir)


# ==============================================================================
# 深度测试用例集
# ==============================================================================

def test_aho_corasick_longest_match(tokenizer):
    """【测试 1】Aho-Corasick 最左最长匹配与重叠消解"""
    text = "执行 <tool_call> 测试 </tool>"

    # 允许所有特殊字符解析
    encoding = tokenizer.encode(text, allowed_special=["all"])
    tokens = encoding.tokens

    assert "<tool_call>" in tokens, "Aho-Corasick 丢失了最长匹配！"

    # 获取 <tool_call> 出现的位置前后的 token，确保它没有被错误拆分
    tool_call_idx = tokens.index("<tool_call>")
    assert "<tool>" not in tokens[:tool_call_idx+1], "发生致命错误：子串 <tool> 提前截断了 <tool_call>！"

def test_special_token_all_collision(tokenizer):
    """【测试 2】防注入系统对 'all' 关键字的安全性校验"""
    text = "请解析 all 和 <tool>"

    # 因为传入了 ["all"]，底层判定为允许全部，所以 <tool> 必须存活
    encoding = tokenizer.encode(text, allowed_special=["all"])
    assert "<tool>" in encoding.tokens

    # 反向安全测试：如果没有授权，恶意注入的 <tool> 必须被切碎
    safe_encoding = tokenizer.encode(text, allowed_special=None)
    assert "<tool>" not in safe_encoding.tokens, "防注入屏障被穿透！"

def test_pre_tokenizer_512_boundary_cut(tokenizer):
    """【测试 3】512 字节分块时的极限 UTF-8 边界切断保护"""
    # '龘' 的 UTF-8 是 3 字节。170 个是 510 字节。171 个是 513 字节。
    # Rust 引擎如果粗暴按字节截断，此处必导致 UTF-8 解析崩溃。
    extreme_text = "龘" * 171

    encoding = tokenizer.encode(extreme_text)
    decoded = tokenizer.decode(encoding.ids)

    assert decoded == extreme_text, "512字节截断算法在处理 UTF-8 中文字符边界时产生了破损！"

def test_sonic_rs_toxic_data_handling():
    """【测试 4】动态 JSONL 榨取引擎对脏数据的吞咽能力"""
    toxic_data = [
        '{"text": "正常文本", "meta": {"id": 1}}',
        '{"text": null, "meta": null}',
        '{"text": ["数组", "代替", "字符串"], "meta": 3.14}',
        '{"text":',
        '{}',
    ]

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("\n".join(toxic_data))
        tmp_path = f.name

    try:
        schema = {
            "extract_text": ("text", "str"),
            "extract_meta": ("meta", "raw_json")
        }
        reader = JsonlFileReader.dynamic(tmp_path, schema)

        t_config = TrainerConfig(vocab_size=260).with_min_frequency(1)
        trainer = BBPETrainer(trainer_config=t_config)

        try:
            trainer.train(reader)
        except Exception as e:
            pytest.fail(f"Rust jsonl 动态解析引擎崩溃，未能接住脏数据异常: {e}")

    finally:
        os.remove(tmp_path)

def test_trainer_exhaustion():
    """【测试 5】小语料大词表要求下的早停死循环检测"""
    tiny_corpus = "A B C D E"

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(tiny_corpus)
        tmp_path = f.name

    try:
        t_config = TrainerConfig(vocab_size=5000).with_min_frequency(1)
        trainer = BBPETrainer(trainer_config=t_config)
        reader = TextFileReader(tmp_path)

        # 必须能够探测到频次耗尽并安全退出
        trainer.train(reader)
    finally:
        os.remove(tmp_path)

def test_fsm_properties_healing(tokenizer):
    """【测试 6】底层 FSM 状态机的查询与容错行为 (对齐 GPU Logits Processor)"""
    core = tokenizer._core

    # 【核心修复】：直接测试底层的基础字节行为，绝对稳定可靠

    # 1. 0xE4 (十进制 228)：这是中文字符的起始字节 (1110xxxx)，它悬空且需要等待 2 个延续字节
    p, h = core.get_utf8_token_props(228)
    assert p == 0 and h == 2, f"起始字节属性计算错误: p={p}, h={h}"

    # 2. 0xB8 (十进制 184)：这是延续字节 (10xxxxxx)，它本身提供 1 个延续单位
    p, h = core.get_utf8_token_props(184)
    assert p == 1 and h == 0, f"延续字节属性计算错误: p={p}, h={h}"

    # 3. 0x20 (十进制 32，空格)：纯 ASCII，不悬空不提供
    p, h = core.get_utf8_token_props(32)
    assert p == 0 and h == 0, f"ASCII字节属性计算错误: p={p}, h={h}"

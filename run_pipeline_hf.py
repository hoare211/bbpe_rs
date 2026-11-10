# run_pipeline_hf.py —— 用 HuggingFace `tokenizers` 训练基线，与 run_pipeline.py
# （本项目自研引擎）做对照。
#
# nit1：原首行注释误写为 `# run_pipeline.py`，与 run_pipeline.py 完全同款，
# 在编辑器标签页 / 日志里极易混淆两个入口。
import os
import sys
import json

# 替换为 HuggingFace 标准库
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

DATA_PATH = 'data/train.jsonl'
# 默认输出目录必须与 run_pipeline.py **分开**。两个脚本都写 `tokenizer.json`，
# 但格式互不兼容（本项目 v1 契约 JSON vs HF tokenizers JSON）：先后运行会互相
# 覆盖，且后跑的脚本可能读到前一个的产物而报出莫名其妙的错。
TOKENIZER_DIR = './tokenizer_output_hf/'
VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36

# ==========================================
# 原版极其复杂的 ChatML 工具调用剧本模板
# ==========================================
MASSIVE_CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"


def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM):
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    additional_tokens_list = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"]

    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens

    print("🚀 配置底层 HuggingFace Tokenizer 引擎...")

    # 1. 初始化 BPE 模型，并装载 ByteLevel Pre-tokenizer/Decoder (为了支持未知字符及表情符号流式解码)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 算出基准词表结束的位置 (与原逻辑一致：保留位给特殊字符续接)
    base_vocab_size = vocab_size - len(all_special_tokens)

    # 2. 声明算法训练目标
    trainer = trainers.BpeTrainer(
        vocab_size=base_vocab_size,
        min_frequency=50,
        special_tokens=[] # 暂不填入特殊字符，等训练完再添加到尾部，以实现原逻辑中的 ID 完美续接
    )

    # 3. 极速 JSONL 解析生成器 (替代原本的 static_text)
    #
    # m17：原实现有两处会把非文本内容喂进 BPE，且完全静默：
    #   ① `data.get('text', line.strip())` —— 缺 text 字段时**退回整行原始 JSON**，
    #      于是 `{`、`"`、`:` 这些 JSON 语法字符被当成语料统计进词表；
    #   ② `except: yield line.strip()` —— 坏行同样被原文塞进语料。
    # 两者都会污染词表与压缩率评估，却没有任何计数/提示。
    # 现在：坏行与缺字段行一律**跳过并计数**，结束后打印占比。
    skipped = {"empty": 0, "bad_json": 0, "no_text": 0, "ok": 0}

    def jsonl_iterator(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    skipped["empty"] += 1
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    skipped["bad_json"] += 1
                    continue
                # 假设你的原始数据里文本保存在 'text' 字段，如果不是请修改
                text = data.get('text')
                if not isinstance(text, str) or not text.strip():
                    skipped["no_text"] += 1
                    continue
                skipped["ok"] += 1
                yield text

    print("⚡ 启动 HuggingFace 底层 BPE Trainer (Rust 高效实现)...")
    tokenizer.train_from_iterator(jsonl_iterator(data_path), trainer=trainer)

    # m17：把被跳过的行讲清楚（此前完全静默）
    total = sum(skipped.values())
    bad = skipped["bad_json"] + skipped["no_text"]
    print(
        f"   语料读取: 有效 {skipped['ok']} 行 | 坏 JSON {skipped['bad_json']} 行 | "
        f"缺 text {skipped['no_text']} 行 | 空行 {skipped['empty']} 行"
    )
    if total and bad:
        print(f"   ⚠️  {bad} 行被跳过（占比 {bad / total * 100:.4f}%）；"
              f"若占比过高请检查数据字段名是否为 'text'")

    # 4. 训练完毕后，手动将特殊字符注入到词表末尾，完美续接原版逻辑中的 AddedTokens
    tokenizer.add_special_tokens(all_special_tokens)

    # 5. 生成规范的 Fast Tokenizer 并封装 Chat Template
    os.makedirs(tokenizer_dir, exist_ok=True)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        chat_template=MASSIVE_CHAT_TEMPLATE,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>"
    )

    # 保存全套配置 (自动生成 tokenizer.json, tokenizer_config.json)
    fast_tokenizer.save_pretrained(tokenizer_dir)
    print("Tokenizer training completed.")


def eval_tokenizer(tokenizer_dir):
    print("🔄 正在加载封装好的 PreTrainedTokenizerFast...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自月球'},
        {"role": "user", "content": '你到底来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]

    # 1. 验证 Chat Template 渲染引擎 (Jinja2)
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print('-'*100)
    print(new_prompt)
    print('-'*100)

    # 2. 验证高速编码引擎 (自动拦截和识别 Special Tokens)
    # HF 原生 encode 会自动解析已注册的特殊字符
    input_ids = tokenizer.encode(new_prompt, add_special_tokens=False)
    print('encoder长度：', len(input_ids))

    # 3. 验证解码一致性
    response = tokenizer.decode(input_ids)
    print('decoder一致性：', response == new_prompt, "\n")
    print('-'*100)

    # 4. 压缩率测试
    print('压缩率测试（Chars/Tokens）：')
    test_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器...",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心...",
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data...",
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science..."
    ]

    total_compression = 0
    for i, text in enumerate(test_texts):
        # 统计文本长度和 Token 数
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        char_count = len(text)
        compression_ratio = char_count / token_count if token_count > 0 else 0
        total_compression += compression_ratio
        print(f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")

    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")
    print('-'*100)

    # 5. 流式解码乱码鲁棒性测试
    print('流式解码（字节缓冲）测试：')
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        # HF 的 ByteLevel Decode 在遇到被截断的 UTF-8 残缺字节时也会输出 \ufffd ()
        current_decode = tokenizer.decode(token_cache, skip_special_tokens=False)

        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens_str = tokenizer.decode(token_cache if isinstance(token_cache, list) else [token_cache], skip_special_tokens=False)
            print(f'Token ID: {str(display_ids):15} -> Raw: {raw_tokens_str:20} -> Decode Str: {current_decode}')
            token_cache = []


if __name__ == '__main__':
    # m17：数据集缺失必须以**非零退出码**结束。原实现只打印警告然后正常退出，
    # CI 会把它判为成功 —— 一个"什么都没训"的绿色流水线比失败更危险。
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到数据集 {DATA_PATH}，终止。", file=sys.stderr)
        print("   若只是想试流程，请创建一个 jsonl 文件；"
              "或在 CI 中显式准备数据后再运行。", file=sys.stderr)
        sys.exit(1)

    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
# run_pipeline.py
import os
import sys
import json

# 导入我们自己用 Rust 和 Python 混合打造的企业级包
from bbpe_tokenizer import (
    BBTokenizer,
    TokenizerGlobalConfig,
    BBPETrainer,
    TrainerConfig,         # 新增：算法域配置
    RuntimeConfig,         # 新增：系统域配置
    JsonlFileReader,       # 新增：极速 JSONL 解析器
    init_rust_logger
)

DATA_PATH = 'data/train.jsonl'
TOKENIZER_DIR = './tokenizer_output/'
VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36

# ==========================================
# 原版极其复杂的 ChatML 工具调用剧本模板
# ==========================================
MASSIVE_CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM):
    # 开启 Rust 底层的 tracing 日志输出
    init_rust_logger()

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

    print("🚀 配置底层 Rust 引擎...")

    # 1. 声明算法训练目标
    t_config = TrainerConfig(vocab_size=vocab_size) \
        .with_min_frequency(50) \
        .with_special_tokens(all_special_tokens)

    # 2. 声明底层系统运行配置 (全默认即可，也可以通过 with_batch_size 等调优)
#     r_config = RuntimeConfig()
#     trainer = BBPETrainer(trainer_config=t_config, runtime_config=r_config)
    trainer = BBPETrainer(trainer_config=t_config)

    # 3. 极其优雅：直接将底层 JSONL 解析器指向数据文件，免去一切临时文件生成！
    # （注：Rust 速度极快，原版中的 10000 行限制被移除，它会瞬间跑完整个文件）
#     reader = JsonlFileReader.static_conversations(data_path)
    reader = JsonlFileReader.static_text(data_path)

    print("⚡ 启动 Rust 底层 BBPETrainer (多线程与 SIMD 并发全开)...")
    trainer.train(reader)

    # 4. 保存基准词表模型
    os.makedirs(tokenizer_dir, exist_ok=True)
    model_path = os.path.join(tokenizer_dir, "tokenizer.json")
    trainer.save(model_path)

    # 5. 校验 Special Tokens（m16）
    #
    # 原实现在这里按 `6400 - 36 = 6364` 重新算 base_vocab_size 并**二次注入**
    # added_tokens。但 Rust 侧 `BBPETrainer::new` 在构造时就已经注入了 special
    # tokens（id = vocab_size - n_special + i），`save` 也会把它写进
    # `added_tokens` ⟹ 这次注入纯属冗余。
    # 真正的问题是**耦合**：id 布局被硬编码在 Python 侧，一旦 vocab_size 或
    # specials 数量变化，Rust 侧布局随之变化而 Python 侧不动，就会静默错位
    # （special token 覆盖已学 token / 反之）。现在改为从模型自身读取并断言
    # 一致 —— 把"静默错位"变成"显式失败"。
    with open(model_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    added_tokens = tokenizer_data.get("added_tokens")
    if not added_tokens:
        raise RuntimeError(
            f"模型 {model_path} 缺少 added_tokens：special tokens 应由 Rust trainer "
            "在构造时注入并随模型写出。请检查 TrainerConfig.with_special_tokens 是否生效。"
        )

    # id 布局的唯一真源是训练配置，此处仅做一致性断言（不再写回）
    base_vocab_size = vocab_size - len(all_special_tokens)
    expected = {t: base_vocab_size + i for i, t in enumerate(all_special_tokens)}
    if added_tokens != expected:
        raise RuntimeError(
            "added_tokens 的 id 布局与训练配置不符（special id 布局发生漂移）：\n"
            f"  模型实际: {added_tokens}\n"
            f"  预期:     {expected}\n"
            "这会导致 special token 与已学 token 错位，必须终止而非静默继续。"
        )
    print(f"✅ added_tokens 校验通过：{len(added_tokens)} 个 special token，"
          f"id {min(added_tokens.values())}–{max(added_tokens.values())}")

    # 6. 生成规范的 tokenizer_config.json (SSOT 全局配置中心)
    config = TokenizerGlobalConfig()
    config.chat_template = MASSIVE_CHAT_TEMPLATE

    config_dict = config.to_dict()
    config_dict["added_tokens_decoder"] = {
        str(base_vocab_size + i): {"content": t, "lstrip": False, "rstrip": False, "single_word": False, "special": True}
        for i, t in enumerate(all_special_tokens)
    }
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)

    print("Tokenizer training completed.")

def eval_tokenizer(tokenizer_dir):
    print("🔄 正在加载封装好的 BBTokenizer...")
    tokenizer = BBTokenizer.from_pretrained(tokenizer_dir)

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自月球'},
        {"role": "user", "content": '你到底来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]

    # 1. 验证 Chat Template 渲染引擎 (Python Jinja2 提供)
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print('-'*100)
    print(new_prompt)
    print('-'*100)

    # 2. 验证防注入与高速编码引擎 (Rust AhoCorasick + BPE 提供)
    # 【注】这里显式允许解析所有的特殊控制字符！
    encoding = tokenizer.encode(new_prompt, allowed_special=["all"])
    input_ids = encoding.ids
    print('encoder长度：', len(input_ids))

    # 3. 验证解码一致性 (Rust 提供)
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
        # 普通文本绝不允许解析任何特殊字符 (allowed_special=None 是我们的最高安全防线)
        token_count = len(tokenizer.encode(text, allowed_special=None).ids)
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
        # Rust 底层通过 String::from_utf8_lossy 优雅处理残破字节，返回 \ufffd
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens_str = tokenizer.decode(token_cache if isinstance(token_cache, list) else [token_cache])
            print(f'Token ID: {str(display_ids):15} -> Raw: {raw_tokens_str:20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    # 确保数据集路径正确。如果你本地没有，可以随便建一个包含中文的 jsonl 或 txt 供测试
    # R4-28：数据集缺失必须**非零退出**。原实现只打印一条警告就正常结束，
    # CI / 批处理会把它当成成功（"什么都没训"的绿色流水线比失败更危险）。
    # 与 run_pipeline_hf.py 的 m17 修法保持同一语义。
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到数据集 {DATA_PATH}，终止。", file=sys.stderr)
        print("   若只是想试流程，请先创建一个含中文的 jsonl/txt 文件，"
              "或修改本文件的 DATA_PATH。", file=sys.stderr)
        sys.exit(1)
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
import torch
from transformers import LogitsProcessor

# 导入局部核心类以添加类型提示
from .core import BBTokenizer

class UTF8StrictLogitsProcessor(LogitsProcessor):
    """
    依靠底层 Rust 预计算的 O(1) 状态机，在 GPU 侧掩盖非法 UTF-8 字节。
    大幅降低流式输出中的中文字节断裂/乱码概率。

    R4-17（保证边界，勿夸大）：本处理器保证的是 —— 在掩码全程生效、没有
    其它 LogitsProcessor 绕过它的前提下，生成序列不会出现"悬空的多字节
    前导字节"（每个字符的延续字节数必须被后续 token 补齐）。Rust 侧 FSM
    已把 C0/C1（overlong）与 F5–F7（超出 U+10FFFF）排除出合法前导字节，
    因此这些非法序列不会被放行。但它**不是**完整的 UTF-8 校验器：
    surrogate / 超范围标量等形态取决于词表内容，请以 decode 结果为准。

    命名说明（第五轮收敛）：类名中的 "Strict" 是历史名称，实际保证是
    **前导字节级**（非法前导不开启悬空 + 悬空必须补齐），而非序列级
    完整 UTF-8 合法性。为兼容已有 API 保留原名；所有文档声明均按此口径。
    """
    def __init__(self, tokenizer: BBTokenizer, batch_size: int = 1, device: str = "cuda"):
        # 直接拿到 Rust 底层的引用
        self.tokenizer = tokenizer._core
        # m18：原先 `pending_bytes` 按构造时的 batch_size 一次性定长，而 __call__
        # 用的是**实际** batch（`input_ids.shape[0]`）。动态 batch 一旦大于初值
        # 就会 IndexError 崩溃。这里只作为初始容量，__call__ 会按需扩容。
        self.pending_bytes = [0] * max(1, batch_size)

        # 【性能核弹】：初始化时将 4 个合法状态数组全部推入 GPU 显存
        self.valid_masks = {
            i: torch.tensor(
                self.tokenizer.get_valid_continuations(i),
                dtype=torch.long,
                device=device
            )
            for i in range(4)
        }

        # m18：状态机被强行突破的次数。正常掩码生效时应恒为 0；
        # 若持续增长，说明有别的 LogitsProcessor 绕过了本掩码，需排查。
        self.violations = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        # m18：按实际 batch 扩容，避免动态 batch 越界崩溃
        if len(self.pending_bytes) < batch_size:
            self.pending_bytes.extend([0] * (batch_size - len(self.pending_bytes)))

        for i in range(batch_size):
            last_token = input_ids[i, -1].item()

            # 1. 极速状态转移 (O(1) 查询 Rust 提供的信息)
            provides, hangs = self.tokenizer.get_utf8_token_props(last_token)

            # 状态机转移。三分支与 Rust 侧 is_valid_for_pending 的判定一致：
            #   provides == pending  ⟹ 恰好补齐前序悬空，转入本 token 自身的 hangs
            #   provides <  pending  ⟹ 只补齐了一部分，剩余待补 = pending - provides
            #   provides >  pending  ⟹ 非法状态（掩码本应拦住），只能硬重置
            if provides == self.pending_bytes[i]:
                self.pending_bytes[i] = hangs
            elif provides < self.pending_bytes[i]:
                # 边界保护：防止被其他处理器干扰时跌入负数
                self.pending_bytes[i] = max(0, self.pending_bytes[i] - provides)
            else:
                # m18：强行突破状态机的生成（如幻觉 / 其他处理器绕过了掩码）。
                # 这里属于**不可恢复的非法状态**：多出的 provides - pending 个延续
                # 字节无处安放，只能丢弃并重置为本 token 自身的悬空数。
                # 此前该分支没有任何标记，静默吞掉了异常；现在至少留下计数，
                # 便于判断"掩码是否真的生效"。
                self.pending_bytes[i] = hangs
                self.violations += 1

            # 2. 物理阉割非法预测 (GPU 张量内极速替换)
            current_pending = self.pending_bytes[i]

            mask = torch.ones_like(scores[i], dtype=torch.bool)

            # 拿到合法 ID 的索引，将这些位置设为 False (放行)
            valid_idx = self.valid_masks.get(current_pending, self.valid_masks[0])
            mask[valid_idx] = False

            # 将被拉黑的非法词概率变为负无穷
            scores[i, mask] = -float('inf')

        return scores

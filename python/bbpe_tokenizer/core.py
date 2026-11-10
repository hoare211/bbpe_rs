import json
import os
import jinja2
from jinja2.sandbox import SandboxedEnvironment
from typing import List, Dict, Union, Optional

# 导入局部的配置类
from .config import TokenizerGlobalConfig
# 导入通过 Maturin 编译打包进来的 Rust 底层引擎
from . import bbpe_rs

class BBTokenizer:
    def __init__(self, core_tokenizer: bbpe_rs.BBPETokenizer, config: TokenizerGlobalConfig):
        self._core = core_tokenizer
        self.config = config

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "BBTokenizer":
        """从文件夹加载模型和配置（JSON 格式）"""
        json_path = os.path.join(model_dir, "tokenizer.json")
        config_path = os.path.join(model_dir, "tokenizer_config.json")

        # M6（重要修复）：原实现优先检测 `tokenizer.bbpe`，并打印
        # "⚡ 启动 Mmap 零拷贝加载"，随后把该路径交给 Rust 侧
        # `bbpe_rs.BBPETokenizer(path)`。但 Rust 端 `BBPETokenizer::load` 只有
        # `serde_json::from_reader` 一条路径，**不存在任何二进制/mmap 加载实现**
        # （已全量搜索确认）。因此该分支是死功能：一旦磁盘上出现 .bbpe 文件
        # （用户手建、改名、或未来产物），加载会在 JSON 解析阶段直接失败。
        # 声称支持却不支持，比不支持更糟 —— 在 Rust 侧真正实现二进制序列化
        # 之前，这里只认 JSON。
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"在 {model_dir} 中找不到 tokenizer.json")
        model_path = json_path

        # 加载 Rust 底层引擎
        core_tokenizer = bbpe_rs.BBPETokenizer(model_path)

        # 加载配置
        config = TokenizerGlobalConfig()
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
                for k, v in cfg_data.items():
                    if hasattr(config, k):
                        setattr(config, k, v)

        return cls(core_tokenizer, config)

    def save_pretrained(self, save_dir: str):
        """保存模型核心文件与配置供下游使用 (完美填坑)

        注：产物固定为 `tokenizer.json`（JSON 格式）。当前 Rust 引擎仅支持
        JSON 加载，不存在二进制缓存格式（见 from_pretrained 的 M6 说明）。
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. 写入 Python 侧的业务配置 (Jinja 模板, 特殊词映射等)
        config_path = os.path.join(save_dir, "tokenizer_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4, ensure_ascii=False)

        # 2. 调用刚刚贯通的底层引擎，导出核心 BPE 数据！
        model_path = os.path.join(save_dir, "tokenizer.json")
        self._core.save(model_path)

    def encode(self, text: str, allowed_special: Optional[List[str]] = None):
        """
        全功能编码方法：
        返回包含 ids, tokens, offsets 的富文本对象。
        默认拒绝对一切特殊字符的解析 (防注入最高防御)。
        """
        return self._core.encode(text, allowed_special=allowed_special)

    # 对齐 Tiktoken 的极速编码接口
    def encode_ordinary(self, text: str) -> List[int]:
        """
        极速编码方法：
        完全忽略任何特殊字符解析，不返回 offsets 对齐信息，仅返回 Token ID 列表。
        适用于海量纯文本的安全、极速分词。

        R4-4：预分词失败（自定义 pattern 未覆盖全部文本）时抛 RuntimeError，
        与 encode 同口径 —— 不再返回空/不完整结果且仅 stderr 告警。
        """
        return self._core.encode_ordinary(text)

    def encode_parallel(self, text: str, allowed_special: Optional[List[str]] = None, num_threads: Optional[int] = None):
        """
        [长文本专用] 多线程全功能编码方法：
        利用底层 Rayon 线程池加速万字以上超长文本的分词。
        返回包含 ids, tokens, offsets 的富文本对象。
        *警告：请勿在依赖 os.fork() 的多进程环境 (如 PyTorch DataLoader / Celery) 中盲目调用，以防线程池死锁。
        """
        return self._core.encode_parallel(text, allowed_special=allowed_special, num_threads=num_threads)

    def encode_ordinary_parallel(self, text: str, num_threads: Optional[int] = None) -> List[int]:
        """
        [长文本专用] 多线程极速编码方法：
        利用底层 Rayon 线程池，完全忽略特殊字符解析，不返回 offsets 对齐信息，仅返回 Token ID 列表。
        适用于海量纯文本 (如几十万字小说) 的极致安全、极速分词。

        *警告：请勿在依赖 os.fork() 的多进程环境 (如 PyTorch DataLoader / Celery) 中盲目调用，以防线程池死锁。
         （本方法与上方的 encode_parallel 一样使用 Rayon 全局线程池，该警告此前只写在后者身上。）

        R4-4：预分词失败时抛 RuntimeError（与 encode_parallel 同口径）。
        """
        return self._core.encode_ordinary_parallel(text, num_threads=num_threads)

    def decode(self, ids: List[int]) -> str:
        """基础解码方法"""
        return self._core.decode(ids)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = True
    ) -> Union[str, List[int]]:
        """推理侧绝对主力：提供 Jinja2 渲染"""
        # R4-19：chat_template 来自模型目录（`tokenizer_config.json`），对
        # 不可信模型目录而言就是**不可信输入**。普通 `jinja2.Environment` 允许
        # 模板访问任意属性 / 调用任意方法（`''.__class__.__mro__[1]...` 式
        # 沙箱逃逸），`apply_chat_template` 于是构成服务端模板注入面。
        # 改用 `SandboxedEnvironment`：属性访问与可调用对象受限，
        # 与 transformers 自身渲染 chat template 的做法一致。
        env = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda obj: json.dumps(obj, ensure_ascii=False)

        template = env.from_string(self.config.chat_template)
        rendered = template.render(messages=messages, add_generation_prompt=add_generation_prompt)

        if tokenize:
            # 推理端自己拼好的模板，我们允许解析里面的 special token
            return self.encode(rendered, allowed_special=["all"]).ids
        return rendered

    def _special_id(self, token: str, what: str) -> int:
        """R4-18：取 special token 的**唯一** id，未注册/多 token 时显式失败。

        原实现 `self.encode(bos, allowed_special=["all"]).ids[0]` 有两个静默
        失败形态：① 配置里的 BOS/EOS 未注册为 special 时，它会被编码成**多个
        普通 token**，只取首 ID、其余静默丢弃（拼接出的 prompt 边界全错）；
        ② 编码结果为空时 `IndexError`（与"配置错误"毫无关联的报错）。
        这里要求"恰好一个 token"且它是 special，否则抛出可操作的错误。
        """
        ids = self.encode(token, allowed_special=["all"]).ids
        if len(ids) != 1:
            raise ValueError(
                f"{what} {token!r} 未注册为 special token：编码得到 {len(ids)} 个 token "
                f"（期望恰好 1 个）。请把 {token!r} 加入 TrainerConfig.with_special_tokens "
                f"并重新训练，或在 tokenizer_config.json 中修正。"
            )
        return ids[0]

    def encode_dialogue_safely(self, messages: List[Dict[str, str]]) -> Dict[str, List[int]]:
        """
        【大厂核心机密】训练侧 (SFT) 绝对主力：
        绕过字符串拼接带来的注入风险，自动生成 labels (-100)。
        """
        # 特殊字符需要被正确解析为单个 ID，因此开启 ["all"]
        # R4-18：BOS/EOS 必须恰好对应一个 special ID，否则显式失败
        bos_id = self._special_id(self.config.bos_token, "bos_token")
        eos_id = self._special_id(self.config.eos_token, "eos_token")
        nl_id = self.encode("\n", allowed_special=["all"]).ids[0]

        input_ids = []
        labels = []

        for msg in messages:
            role_ids = [bos_id] + self.encode(msg["role"], allowed_special=["all"]).ids + [nl_id]

            # 【性能与安全双丰收】：这里正好换成极速的 encode_ordinary！
            # 用户的 content 绝对不需要解析出 offsets，且绝对不允许解析特殊字符
            content_ids = self.encode_ordinary(msg["content"])

            tail_ids = [eos_id, nl_id]
            turn_ids = role_ids + content_ids + tail_ids

            input_ids.extend(turn_ids)

            # 构建 PyTorch 训练所需的 Label 掩码
            if msg["role"] == self.config.role_assistant:
                turn_labels = [-100] * len(role_ids) + content_ids + tail_ids
            else:
                turn_labels = [-100] * len(turn_ids)

            labels.extend(turn_labels)

        return {"input_ids": input_ids, "labels": labels}

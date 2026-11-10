from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TokenizerGlobalConfig:
    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|endoftext|>"
    unk_token: str = "<|endoftext|>"

    role_system: str = "system"
    role_user: str = "user"
    role_assistant: str = "assistant"

    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    think_start: str = "<think>"
    think_end: str = "</think>"

    model_max_length: int = 131072

    # 新增一个隐藏字段，用来保存外部强行覆盖的模板
    _custom_chat_template: Optional[str] = None

    @property
    def chat_template(self) -> str:
        """如果外部赋了值，就用外部的；否则动态生成。"""
        if self._custom_chat_template is not None:
            return self._custom_chat_template

        return f"""{{%- for message in messages %}}
    {{{{ '{self.bos_token}' + message['role'] + '\\n' + message['content'] + '{self.eos_token}\\n' }}}}
{{%- endfor %}}
{{%- if add_generation_prompt %}}
    {{{{ '{self.bos_token}{self.role_assistant}\\n' }}}}
{{%- endif %}}"""

    @chat_template.setter
    def chat_template(self, value: str):
        """允许通过 setattr 或 json 读取来覆盖模板"""
        self._custom_chat_template = value

    def to_dict(self) -> Dict[str, Any]:
        # ... 保持不变
        return {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "model_max_length": self.model_max_length,
            "clean_up_tokenization_spaces": False,
            "add_bos_token": False,
            "add_eos_token": False,
            "chat_template": self.chat_template,
            "tokenizer_class": "PreTrainedTokenizerFast"
        }

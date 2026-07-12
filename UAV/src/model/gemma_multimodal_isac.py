"""
BEV-image 分支的 Gemma3 多模态 UAV-ISAC 模型封装。

这是当前最小可用的模型侧桥接层：
  prompt + image -> Gemma3ForConditionalGeneration -> control states
  -> ConstraintProjectionHead -> delta_q / delta_a / delta_p
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .projection_head import ConstraintProjectionHead


def _first_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Gemma3MultimodalISAC(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "google/gemma-3-4b-it",
        use_4bit: bool = True,
        num_control_tokens: int = 8,
        proj_head_config: Optional[Dict] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        freeze_vision_tower: bool = True,
        enable_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
    ):
        super().__init__()

        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        if use_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.base_model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        self.num_control_tokens = num_control_tokens
        # 控制 token 只作为控制头读出的锚点，不参与普通自然语言生成目标。
        control_tokens = [f"<ctrl_{i}>" for i in range(num_control_tokens)]
        num_added = self.tokenizer.add_tokens(control_tokens, special_tokens=True)
        if num_added > 0:
            self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.control_token_ids = self.tokenizer.convert_tokens_to_ids(control_tokens)

        self.lora_enabled = enable_lora
        if enable_lora:
            # LoRA 烟雾测试用于确认 backbone 可训练链路；默认训练仍保持只训练投影头。
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            if use_4bit:
                from peft import prepare_model_for_kbit_training

                self.base_model = prepare_model_for_kbit_training(self.base_model)
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.base_model = get_peft_model(self.base_model, peft_config)

        config = self.base_model.config
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_dim = config.text_config.hidden_size
        elif hasattr(config, "hidden_size"):
            hidden_dim = config.hidden_size
        else:
            raise AttributeError("Cannot find hidden_size in Gemma3 multimodal config.")
        self.hidden_dim = hidden_dim

        if freeze_vision_tower:
            vision_tower = getattr(self.base_model, "vision_tower", None)
            if vision_tower is None:
                vision_tower = getattr(self.base_model, "vision_model", None)
            if vision_tower is not None:
                for param in vision_tower.parameters():
                    param.requires_grad = False

        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()

        if proj_head_config is None:
            proj_head_config = {}
        proj_head_config.setdefault("hidden_dim", hidden_dim)
        proj_head_config.setdefault("num_control_tokens", num_control_tokens)
        self.projection_head = ConstraintProjectionHead(**proj_head_config)

    @property
    def device(self) -> torch.device:
        return _first_device(self.base_model)

    def _extract_control_states(
        self,
        hidden_states: torch.Tensor,
        control_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        states = []
        for b in range(batch_size):
            ctrl_positions = control_mask[b].nonzero(as_tuple=True)[0]
            ctrl_hidden = hidden_states[b, ctrl_positions]
            if ctrl_hidden.shape[0] < self.num_control_tokens:
                pad = torch.zeros(
                    self.num_control_tokens - ctrl_hidden.shape[0],
                    self.hidden_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                ctrl_hidden = torch.cat([ctrl_hidden, pad], dim=0)
            elif ctrl_hidden.shape[0] > self.num_control_tokens:
                ctrl_hidden = ctrl_hidden[:self.num_control_tokens]
            states.append(ctrl_hidden)
        return torch.stack(states, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        control_mask: torch.Tensor,
        q_current: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
            # 烟雾测试阶段不使用 token-level CE，尽量少保留 logits 以降低显存峰值。
            "logits_to_keep": 1,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        for key, value in kwargs.items():
            if hasattr(value, "shape"):
                model_inputs[key] = value

        try:
            outputs = self.base_model(**model_inputs)
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            model_inputs.pop("logits_to_keep", None)
            outputs = self.base_model(**model_inputs)
        if getattr(outputs, "hidden_states", None) is None:
            raise RuntimeError("Gemma3 forward did not return hidden_states.")
        last_hidden = outputs.hidden_states[-1]
        control_states = self._extract_control_states(last_hidden, control_mask)

        self.projection_head = self.projection_head.to(control_states.device)
        if q_current is not None:
            q_current = q_current.to(control_states.device)
        prior_hat = self.projection_head(control_states.float(), q_current)

        return {
            "logits": getattr(outputs, "logits", None),
            "hidden_states": last_hidden,
            "control_states": control_states,
            **prior_hat,
        }

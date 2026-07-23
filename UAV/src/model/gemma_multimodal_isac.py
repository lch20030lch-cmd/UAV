"""
BEV-image 分支的 Gemma3 多模态 UAV-ISAC 模型封装。

这是当前最小可用的模型侧桥接层：
  prompt + image -> Gemma3ForConditionalGeneration -> control states
  -> ConstraintProjectionHead -> delta_q / delta_a / delta_p
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .projection_head import ConstraintProjectionHead


def _first_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_vision_parameter_name(name: str) -> bool:
    """Return whether a nested PEFT parameter belongs to the vision tower."""
    return "vision" in name.lower()


def freeze_vision_parameter_tree(module: nn.Module) -> list[str]:
    """Freeze vision parameters after PEFT wrapping, including injected LoRA.

    PEFT nests the original Gemma3 model below several wrappers, so direct
    ``getattr(model, 'vision_tower')`` lookup can miss the actual parameters.
    Matching full parameter paths freezes both base vision weights and every
    vision-side LoRA tensor without relying on wrapper-specific attributes.
    """
    frozen = []
    for name, parameter in module.named_parameters():
        if is_vision_parameter_name(name):
            parameter.requires_grad = False
            frozen.append(name)
    return frozen


def keep_vision_modules_in_eval_mode(module: nn.Module) -> list[str]:
    """Keep a frozen vision tower deterministic during language-side tuning."""
    frozen_modules = []
    for name, child in module.named_modules():
        if name and is_vision_parameter_name(name):
            child.eval()
            frozen_modules.append(name)
    return frozen_modules


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
        lora_checkpoint: Optional[str] = None,
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

        self.lora_enabled = enable_lora or lora_checkpoint is not None
        self.loaded_lora_checkpoint = None
        if self.lora_enabled:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            if lora_checkpoint is not None:
                # 诊断和续跑时直接挂载已保存的 adapter，避免只评估原始 backbone。
                from peft import PeftModel

                lora_path = Path(lora_checkpoint)
                if not lora_path.exists():
                    raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
                self.base_model = PeftModel.from_pretrained(
                    self.base_model,
                    str(lora_path),
                    is_trainable=enable_lora,
                )
                self.loaded_lora_checkpoint = str(lora_path)
            else:
                # LoRA 烟雾测试用于确认 backbone 可训练链路；默认训练仍保持只训练投影头。
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
        input_embeddings = self.base_model.get_input_embeddings()
        self.control_token_offsets = nn.Parameter(
            torch.zeros(
                self.num_control_tokens,
                hidden_dim,
                dtype=torch.float32,
                device=input_embeddings.weight.device,
            )
        )
        self._control_embedding_hook = input_embeddings.register_forward_hook(
            self._inject_control_token_offsets
        )

        self.frozen_vision_parameter_names = []
        if freeze_vision_tower:
            self.frozen_vision_parameter_names = freeze_vision_parameter_tree(
                self.base_model
            )
            if not self.frozen_vision_parameter_names:
                raise RuntimeError(
                    "freeze_vision_tower=True but no nested vision parameters were found"
                )

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

    def _inject_control_token_offsets(self, module, inputs, output):
        """Add a small trainable adapter only at control-token positions."""
        if not inputs or not torch.is_tensor(inputs[0]):
            return output
        token_ids = inputs[0]
        offset_indices = torch.full_like(token_ids, -1)
        for index, token_id in enumerate(self.control_token_ids):
            offset_indices = torch.where(
                token_ids == int(token_id),
                torch.full_like(offset_indices, index),
                offset_indices,
            )
        valid = offset_indices >= 0
        if not torch.any(valid):
            return output
        offsets = self.control_token_offsets[
            offset_indices.clamp_min(0)
        ].to(dtype=output.dtype)
        return output + offsets * valid.unsqueeze(-1).to(output.dtype)

    def save_control_token_embeddings(self, save_dir: str) -> Dict[str, str]:
        """只保存新增控制 token 对应的 embedding 行。"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        embed = self.base_model.get_input_embeddings()
        ctrl_embed = embed.weight.data[self.control_token_ids].detach().clone().cpu()
        ctrl_embed_path = save_path / "ctrl_embed.pt"
        torch.save(ctrl_embed, ctrl_embed_path)

        saved = {"ctrl_embed": str(ctrl_embed_path)}
        ctrl_offset_path = save_path / "ctrl_offset.pt"
        torch.save(
            self.control_token_offsets.detach().clone().cpu(),
            ctrl_offset_path,
        )
        saved["ctrl_offset"] = str(ctrl_offset_path)
        output_embeds = self.base_model.get_output_embeddings()
        if output_embeds is not None and output_embeds.weight.data_ptr() != embed.weight.data_ptr():
            ctrl_lm = output_embeds.weight.data[self.control_token_ids].detach().clone().cpu()
            ctrl_lm_path = save_path / "ctrl_lm_head.pt"
            torch.save(ctrl_lm, ctrl_lm_path)
            saved["ctrl_lm_head"] = str(ctrl_lm_path)
        return saved

    def load_control_token_embeddings(self, load_dir: str) -> Dict[str, str]:
        """恢复 checkpoint 中保存的控制 token embedding 行。"""
        load_path = Path(load_dir)
        ctrl_embed_path = load_path / "ctrl_embed.pt"
        loaded = {}
        if ctrl_embed_path.exists():
            ctrl_embed = torch.load(ctrl_embed_path, map_location="cpu")
            embed = self.base_model.get_input_embeddings()
            expected_shape = (
                self.num_control_tokens,
                int(embed.weight.shape[1]),
            )
            if tuple(ctrl_embed.shape) != expected_shape:
                raise ValueError(
                    "control token embedding shape mismatch: "
                    f"{tuple(ctrl_embed.shape)} != {expected_shape}"
                )
            embed.weight.data[self.control_token_ids] = ctrl_embed.to(
                dtype=embed.weight.dtype,
                device=embed.weight.device,
            )
            loaded["ctrl_embed"] = str(ctrl_embed_path)

        ctrl_offset_path = load_path / "ctrl_offset.pt"
        if ctrl_offset_path.exists():
            ctrl_offset = torch.load(ctrl_offset_path, map_location="cpu")
            if tuple(ctrl_offset.shape) != tuple(
                self.control_token_offsets.shape
            ):
                raise ValueError(
                    "control token offset shape mismatch: "
                    f"{tuple(ctrl_offset.shape)} != "
                    f"{tuple(self.control_token_offsets.shape)}"
                )
            self.control_token_offsets.data.copy_(
                ctrl_offset.to(
                    device=self.control_token_offsets.device,
                    dtype=self.control_token_offsets.dtype,
                )
            )
            loaded["ctrl_offset"] = str(ctrl_offset_path)

        ctrl_lm_path = load_path / "ctrl_lm_head.pt"
        if ctrl_lm_path.exists():
            ctrl_lm = torch.load(ctrl_lm_path, map_location="cpu")
            output_embeds = self.base_model.get_output_embeddings()
            if output_embeds is not None:
                expected_shape = (
                    self.num_control_tokens,
                    int(output_embeds.weight.shape[1]),
                )
                if tuple(ctrl_lm.shape) != expected_shape:
                    raise ValueError(
                        "control LM-head row shape mismatch: "
                        f"{tuple(ctrl_lm.shape)} != {expected_shape}"
                    )
                output_embeds.weight.data[self.control_token_ids] = ctrl_lm.to(
                    dtype=output_embeds.weight.dtype,
                    device=output_embeds.weight.device,
                )
                loaded["ctrl_lm_head"] = str(ctrl_lm_path)
        return loaded

    def _extract_control_states(
        self,
        hidden_states: torch.Tensor,
        control_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        states = []
        for b in range(batch_size):
            ctrl_positions = control_mask[b].nonzero(as_tuple=True)[0]
            if ctrl_positions.numel() != self.num_control_tokens:
                raise ValueError(
                    "control_mask must select exactly "
                    f"{self.num_control_tokens} tokens per sample; "
                    f"sample {b} selects {ctrl_positions.numel()}"
                )
            ctrl_hidden = hidden_states[b, ctrl_positions]
            states.append(ctrl_hidden)
        return torch.stack(states, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        control_mask: torch.Tensor,
        q_current: Optional[torch.Tensor] = None,
        q_geometry_cues: Optional[torch.Tensor] = None,
        q_geometry_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        compute_full_logits: bool = False,
        logits_to_keep: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }
        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep
        elif not compute_full_logits:
            # Control-only stages do not need token-level CE.  Keeping one
            # position materially reduces the vocabulary-logit memory peak.
            model_inputs["logits_to_keep"] = 1
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
        if q_geometry_cues is not None:
            q_geometry_cues = q_geometry_cues.to(control_states.device)
        if q_geometry_mask is not None:
            q_geometry_mask = q_geometry_mask.to(control_states.device)
        prior_hat = self.projection_head(
            control_states.float(),
            q_current,
            q_geometry_cues,
            q_geometry_mask,
        )

        return {
            "logits": getattr(outputs, "logits", None),
            "hidden_states": last_hidden,
            "control_states": control_states,
            **prior_hat,
        }

from .gemma_isac import Gemma3ISAC, ensure_gc_and_freeze_lm_head
from .gemma_multimodal_isac import Gemma3MultimodalISAC
from .projection_head import ConstraintProjectionHead
from .losses import UAVISACLosses


def build_proj_head_config(model_cfg: dict, sim_cfg: dict) -> dict:
    """从 YAML 配置构造 ConstraintProjectionHead 参数字典

    原先在 train_sft.py / train_dpo.py / evaluate.py 中有 3-4 处
    重复的 15 行字典构造代码, 现提取为此工厂函数。
    """
    return {
        "hidden_dim": model_cfg["control_token"]["hidden_dim"],
        "num_control_tokens": model_cfg["control_token"]["num_tokens"],
        "mlp_hidden": model_cfg["projection_head"]["mlp_hidden"],
        "readout_out_dim": model_cfg["projection_head"]["readout_out_dim"],
        "M": sim_cfg["num_uavs"],
        "K": sim_cfg["num_users"],
        "area_w": sim_cfg["area_size"][0],
        "area_h": sim_cfg["area_size"][1],
        "h_min": sim_cfg["altitude_min_m"],
        "h_max": sim_cfg["altitude_max_m"],
        "v_max_dt": sim_cfg["uav_max_speed_ms"] * sim_cfg["slot_duration_s"],
        "p_max": 10 ** ((sim_cfg["p_max_dbm"] - 30) / 10),
        "K_max": sim_cfg["load_cap_per_uav"],
        "tau_power": model_cfg["projection_head"]["tau_power"],
        "tau_assoc": model_cfg["projection_head"]["tau_assoc"],
        "sinkhorn_iters": model_cfg["projection_head"]["sinkhorn_iters"],
        "head_type": model_cfg["projection_head"].get("head_type", "shared"),
        "q_projection_mode": model_cfg["projection_head"].get("q_projection_mode", "clip"),
        "q_geometry_mode": model_cfg["projection_head"].get("q_geometry_mode", "none"),
        "q_fixed_cue_weights": model_cfg["projection_head"].get("q_fixed_cue_weights"),
        "q_residual_max_scale": model_cfg["projection_head"].get("q_residual_max_scale", 1.0),
        "q_residual_gate_init": model_cfg["projection_head"].get("q_residual_gate_init", 0.05),
        "power_assoc_gate_strength": model_cfg["projection_head"].get(
            "power_assoc_gate_strength",
            0.0,
        ),
    }

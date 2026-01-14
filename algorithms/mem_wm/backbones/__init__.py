from typing import Optional, List, Literal
from .cogv import create_cogv_like_model
from .flowm import create_flowm_model
from omegaconf import DictConfig

__all__ = ["create_main_model"]

def create_main_model(
        backbone_cfg: DictConfig,
    ):
    AVAILABLE_MODELS = ["cogv", "hybrid_ssm", "hybrid_ssm_official", "flowm"]
    if backbone_cfg.model_type in ["cogv", "hybrid_ssm", "hybrid_ssm_official"]:
        return create_cogv_like_model(**backbone_cfg)
    elif backbone_cfg.model_type in ["fernn", "flow_vit"]:
        return create_flowm_model(backbone_cfg=backbone_cfg)
    else:
        raise ValueError(f"Invalid model type: {backbone_cfg.model_type}. Available models: {AVAILABLE_MODELS}")



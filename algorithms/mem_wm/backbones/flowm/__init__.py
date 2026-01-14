from .flowm_models_3d import ViTBlockworldModel
from .flowm_models_2d import FlowEquivariantRNN, LinearRNNBaseline  # add LinearRNNBaseline
from omegaconf import DictConfig
from utils.distributed_utils import rank_zero_print
import ast 

AVAILABLE_MODELS = ["flowm", "linear_rnn", "flow_vit"]

__all__ = ["create_flowm_model"]

def create_flowm_model(backbone_cfg: DictConfig):
    if backbone_cfg.name == "fernn": # Need to keep this as fernn due to legacy naming
        model = FlowEquivariantRNN(
            input_channels=backbone_cfg.input_channels,
            hidden_channels=backbone_cfg.hidden_channels,
            world_size=backbone_cfg.world_size,
            window_size=backbone_cfg.window_size,
            output_channels=backbone_cfg.output_channels,
            h_kernel_size=backbone_cfg.h_kernel_size,
            u_kernel_size=backbone_cfg.u_kernel_size,
            decoder_conv_layers=backbone_cfg.decoder_conv_layers,
            use_mlp_decoder=backbone_cfg.use_mlp_decoder,
            use_mlp_encoder=backbone_cfg.use_mlp_encoder,
            cell_type=backbone_cfg.cell_type,
            v_range=backbone_cfg.v_range,
            self_motion_equivariance=not backbone_cfg.no_self_motion_equivariance
        )
    elif backbone_cfg.name == "linear_rnn":
        model = LinearRNNBaseline(input_channels=backbone_cfg.input_channels,
                            hidden_dim=backbone_cfg.hidden_channels,
                            window_size=backbone_cfg.window_size,
                            output_channels=backbone_cfg.input_channels,
                            use_mlp_decoder=backbone_cfg.use_mlp_decoder)
    elif backbone_cfg.name == "flow_vit":
        input_shape = getattr(backbone_cfg, "input_shape", (backbone_cfg.input_channels, 128, 128))
        if isinstance(input_shape, str):
            input_shape = ast.literal_eval(input_shape)
        flow_vit_ablation_params = getattr(backbone_cfg, "flow_vit_ablation_params")
        model = ViTBlockworldModel(
            input_shape=input_shape,
            world_size=backbone_cfg.world_size,
            embed_dim=getattr(backbone_cfg, "hidden_channels", 256),
            img_patch_size=getattr(backbone_cfg, "patch_size", 16),
            map_proc_encoder_depth=getattr(backbone_cfg, "map_proc_encoder_depth", 6),
            dec_depth=getattr(backbone_cfg, "decoder_depth", 6),
            num_heads=getattr(backbone_cfg, "num_heads", 8),
            fov_deg=getattr(backbone_cfg, "fov_deg", 60.0),
            v_range=getattr(backbone_cfg, "v_range", 0),
            v_channel_identity_emb=getattr(flow_vit_ablation_params, "v_channel_identity_emb", False),
            v_channel_maxpool_decode=getattr(flow_vit_ablation_params, "v_channel_maxpool_decode", False),
            v_channel_mixing_mode=getattr(flow_vit_ablation_params, "v_channel_mixing_mode", "3d"),
            no_self_motion_equiv=getattr(flow_vit_ablation_params, "no_self_motion_equiv", False),
            external_cond_dim=getattr(flow_vit_ablation_params, "external_cond_dim", 5),
        )
    else:
        raise ValueError(f"Invalid model type: {backbone_cfg.name}. Available models: {AVAILABLE_MODELS}")

    return model
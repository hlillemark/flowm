from omegaconf import DictConfig

from .rssm import RSSMWorldModel

__all__ = ["RSSMWorldModel", "create_rssm_model"]


def create_rssm_model(backbone_cfg: DictConfig) -> RSSMWorldModel:
    kwargs = dict(
        input_shape=tuple(backbone_cfg.input_shape),
        action_dim=backbone_cfg.action_dim,
        deter=backbone_cfg.deter,
        hidden=backbone_cfg.hidden,
        stoch=backbone_cfg.stoch,
        classes=backbone_cfg.classes,
        blocks=backbone_cfg.blocks,
        obs_embed_dim=backbone_cfg.obs_embed_dim,
        cnn_depth=backbone_cfg.cnn_depth,
        act=backbone_cfg.act,
        norm=backbone_cfg.norm,
        unimix=backbone_cfg.unimix,
        outscale=backbone_cfg.outscale,
        imglayers=backbone_cfg.imglayers,
        obslayers=backbone_cfg.obslayers,
        dynlayers=backbone_cfg.dynlayers,
        absolute=backbone_cfg.absolute,
        free_nats=backbone_cfg.free_nats,
        action_embed_dim=backbone_cfg.action_embed_dim,
        loss_scale_recon=backbone_cfg.loss_scale_recon,
        loss_scale_dyn=backbone_cfg.loss_scale_dyn,
        loss_scale_rep=backbone_cfg.loss_scale_rep,
    )
    return RSSMWorldModel(**kwargs)

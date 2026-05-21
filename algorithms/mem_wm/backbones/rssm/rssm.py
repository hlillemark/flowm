import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, out_channels),
            _build_activation(act),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DeconvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(1, out_channels),
            _build_activation(act),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, act: str, outscale: float = 1.0
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("MLP layers must be >= 1")
        modules = []
        cur_dim = in_dim
        for _ in range(layers - 1):
            modules.extend([nn.Linear(cur_dim, hidden_dim), RMSNorm(hidden_dim), _build_activation(act)])
            cur_dim = hidden_dim
        modules.append(nn.Linear(cur_dim, out_dim))
        self.net = nn.Sequential(*modules)
        self.outscale = outscale
        self._init_last()

    def _init_last(self) -> None:
        last = None
        for module in reversed(self.net):
            if isinstance(module, nn.Linear):
                last = module
                break
        if last is not None and self.outscale != 1.0:
            with torch.no_grad():
                last.weight.mul_(self.outscale)
                if last.bias is not None:
                    last.bias.mul_(self.outscale)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BlockLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, groups: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(groups, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(groups, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("bgi,goi->bgo", x, self.weight) + self.bias.unsqueeze(0)


@dataclass
class RSSMState:
    deter: Tensor
    stoch: Tensor


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        cnn_depth: int,
        embed_dim: int,
        act: str,
    ):
        super().__init__()
        channels = [cnn_depth, cnn_depth * 2, cnn_depth * 4, cnn_depth * 8]
        in_channels = input_shape[0]
        blocks = []
        for out_channels in channels:
            blocks.append(ConvNormAct(in_channels, out_channels, act))
            in_channels = out_channels
        self.conv = nn.Sequential(*blocks)
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy)
        self.conv_shape = conv_out.shape[1:]
        self.out_dim = int(math.prod(self.conv_shape))
        self.proj = nn.Linear(self.out_dim, embed_dim)

    def forward(self, video: Tensor) -> Tensor:
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        x = self.conv(x)
        x = x.reshape(b * t, -1)
        x = self.proj(x)
        return x.reshape(b, t, -1)


class ConvDecoder(nn.Module):
    def __init__(
        self,
        output_shape: Tuple[int, int, int],
        deter_dim: int,
        stoch: int,
        classes: int,
        cnn_depth: int,
        act: str,
        bspace: int,
    ):
        super().__init__()
        if output_shape[1] % 16 != 0 or output_shape[2] % 16 != 0:
            raise ValueError(
                f"Decoder expects spatial dims divisible by 16, got {output_shape[1:]}"
            )
        channels = [cnn_depth * 8, cnn_depth * 4, cnn_depth * 2, cnn_depth]
        seed_h = output_shape[1] // 16
        seed_w = output_shape[2] // 16
        if seed_h < 1 or seed_w < 1:
            raise ValueError(f"Unsupported output shape for decoder: {output_shape}")
        self.seed_h = seed_h
        self.seed_w = seed_w
        self.seed_channels = channels[0]
        self.seed_numel = self.seed_channels * seed_h * seed_w
        self.bspace = bspace
        if deter_dim % bspace != 0 or self.seed_numel % bspace != 0:
            raise ValueError(
                f"deter_dim={deter_dim} and seed_numel={self.seed_numel} must be divisible by bspace={bspace}"
            )
        self.deter_proj = BlockLinear(
            deter_dim // bspace, self.seed_numel // bspace, bspace
        )
        self.stoch_proj_in = nn.Linear(stoch * classes, 2 * channels[0])
        self.stoch_proj_norm = RMSNorm(2 * channels[0])
        self.stoch_proj_out = nn.Linear(2 * channels[0], self.seed_numel)
        self.seed_norm = RMSNorm(self.seed_numel)
        self.seed_act = _build_activation(act)
        self.up = nn.Sequential(
            DeconvNormAct(channels[0], channels[1], act),
            DeconvNormAct(channels[1], channels[2], act),
            DeconvNormAct(channels[2], channels[3], act),
        )
        self.out = nn.ConvTranspose2d(
            channels[3], output_shape[0], kernel_size=4, stride=2, padding=1
        )

    def forward(self, deter: Tensor, stoch: Tensor, batch_size: int, time_steps: int) -> Tensor:
        deter_groups = deter.view(batch_size * time_steps, self.bspace, -1)
        x0 = self.deter_proj(deter_groups).reshape(batch_size * time_steps, -1)
        x1 = stoch.reshape(batch_size * time_steps, -1)
        x1 = self.seed_act(self.stoch_proj_norm(self.stoch_proj_in(x1)))
        x1 = self.stoch_proj_out(x1)
        x = self.seed_act(self.seed_norm(x0 + x1))
        x = x.view(batch_size * time_steps, self.seed_channels, self.seed_h, self.seed_w)
        x = self.up(x)
        x = torch.sigmoid(self.out(x))
        _, c, h, w = x.shape
        return x.view(batch_size, time_steps, c, h, w)


class DreamerRSSMCore(nn.Module):
    def __init__(
        self,
        obs_embed_dim: int,
        action_dim: int,
        deter: int,
        hidden: int,
        stoch: int,
        classes: int,
        blocks: int,
        act: str,
        norm: str,
        unimix: float,
        outscale: float,
        imglayers: int,
        obslayers: int,
        dynlayers: int,
        absolute: bool,
        action_embed_dim: int,
    ):
        super().__init__()
        if norm.lower() != "rms":
            raise ValueError(f"Only rms norm is supported right now, got {norm}")
        if deter % blocks != 0:
            raise ValueError(f"deter={deter} must be divisible by blocks={blocks}")
        self.deter = deter
        self.hidden = hidden
        self.stoch = stoch
        self.classes = classes
        self.blocks = blocks
        self.block_size = deter // blocks
        self.absolute = absolute
        self.unimix = unimix
        self.action_dim = action_dim
        self.act = _build_activation(act)

        self.deter_in = nn.Linear(deter, hidden)
        self.stoch_in = nn.Linear(stoch * classes, hidden)
        self.action_in = nn.Linear(action_dim, hidden)
        self.in_norms = nn.ModuleList([RMSNorm(hidden) for _ in range(3)])

        self.dyn_layers = nn.ModuleList()
        self.dyn_norms = nn.ModuleList()
        dyn_in_dim = self.block_size + 3 * hidden
        for _ in range(dynlayers):
            self.dyn_layers.append(BlockLinear(dyn_in_dim, self.block_size, blocks))
            self.dyn_norms.append(RMSNorm(deter))
            dyn_in_dim = self.block_size
        self.gru = BlockLinear(self.block_size, self.block_size * 3, blocks)

        prior_in_dim = deter
        post_in_dim = obs_embed_dim if absolute else obs_embed_dim + deter
        self.prior = MLP(prior_in_dim, hidden, stoch * classes, imglayers, act, outscale=outscale)
        self.posterior = MLP(post_in_dim, hidden, stoch * classes, obslayers, act, outscale=outscale)

    def initial(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RSSMState:
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter, device=device, dtype=dtype),
            stoch=torch.zeros(
                batch_size, self.stoch, self.classes, device=device, dtype=dtype
            ),
        )

    def probs_from_logits(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        if self.unimix > 0:
            probs = probs * (1 - self.unimix) + self.unimix / probs.shape[-1]
        return probs.clamp_min(1e-6)

    def sample_stoch(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        probs = self.probs_from_logits(logits)
        sample_idx = torch.distributions.Categorical(probs=probs).sample()
        sample = F.one_hot(sample_idx, self.classes).to(logits.dtype)
        sample = sample + probs - probs.detach()
        return sample, probs

    def _prior(self, deter: Tensor) -> Tensor:
        return self.prior(deter).view(deter.shape[0], self.stoch, self.classes)

    def _posterior(self, deter: Tensor, obs_embed: Tensor) -> Tensor:
        x = obs_embed if self.absolute else torch.cat([deter, obs_embed], dim=-1)
        return self.posterior(x).view(deter.shape[0], self.stoch, self.classes)

    def _core(self, deter: Tensor, stoch: Tensor, actions: Tensor) -> Tensor:
        stoch_flat = stoch.reshape(stoch.shape[0], -1)
        action_scale = torch.maximum(torch.ones_like(actions), actions.abs()).detach()
        actions = actions / action_scale

        pieces = [self.deter_in(deter), self.stoch_in(stoch_flat), self.action_in(actions)]
        pieces = [self.act(norm(piece)) for piece, norm in zip(pieces, self.in_norms)]
        deter_blocks = deter.view(deter.shape[0], self.blocks, self.block_size)
        mixed = torch.cat(pieces, dim=-1).unsqueeze(1).expand(-1, self.blocks, -1)
        gate_in = torch.cat([deter_blocks, mixed], dim=-1)

        for layer, norms in zip(self.dyn_layers, self.dyn_norms):
            gate_in = layer(gate_in)
            gate_in = gate_in.reshape(deter.shape[0], self.deter)
            gate_in = self.act(norms(gate_in))
            gate_in = gate_in.view(deter.shape[0], self.blocks, self.block_size)

        gates = self.gru(gate_in)
        reset, cand, update = gates.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        deter_next = update * cand + (1.0 - update) * deter_blocks
        return deter_next.reshape(deter.shape[0], self.deter)

    def observe(
        self,
        obs_embed: Tensor,
        actions: Tensor,
        reset: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> Tuple[RSSMState, Dict[str, Tensor]]:
        batch_size, time_steps, _ = obs_embed.shape
        state = self.initial(batch_size, obs_embed.device, obs_embed.dtype)
        deter_seq, stoch_seq, prior_logits_seq, post_logits_seq = [], [], [], []

        if reset is None:
            reset = torch.zeros(batch_size, time_steps, device=obs_embed.device, dtype=torch.bool)
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, time_steps, device=obs_embed.device, dtype=torch.bool)

        for t in range(time_steps):
            prev_state = state
            reset_t = reset[:, t]
            keep = (~reset_t).to(prev_state.deter.dtype).unsqueeze(-1)
            deter_in = prev_state.deter * keep
            stoch_in = prev_state.stoch * keep.unsqueeze(-1)
            action_t = actions[:, t] * keep

            deter = self._core(deter_in, stoch_in, action_t)
            prior_logits = self._prior(deter)
            post_logits = self._posterior(deter, obs_embed[:, t])
            stoch, _ = self.sample_stoch(post_logits)

            valid = valid_mask[:, t].unsqueeze(-1)
            deter = torch.where(valid, deter, prev_state.deter)
            stoch = torch.where(valid.unsqueeze(-1), stoch, prev_state.stoch)
            state = RSSMState(deter=deter, stoch=stoch)

            deter_seq.append(state.deter)
            stoch_seq.append(state.stoch)
            prior_logits_seq.append(prior_logits)
            post_logits_seq.append(post_logits)

        feat = {
            "deter": torch.stack(deter_seq, dim=1),
            "stoch": torch.stack(stoch_seq, dim=1),
            "prior_logits": torch.stack(prior_logits_seq, dim=1),
            "post_logits": torch.stack(post_logits_seq, dim=1),
        }
        return state, feat

    def imagine(
        self,
        actions: Tensor,
        initial_state: RSSMState,
        reset: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size, time_steps = actions.shape[:2]
        state = initial_state
        deter_seq, stoch_seq, prior_logits_seq = [], [], []

        if reset is None:
            reset = torch.zeros(batch_size, time_steps, device=actions.device, dtype=torch.bool)
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, time_steps, device=actions.device, dtype=torch.bool)

        for t in range(time_steps):
            prev_state = state
            reset_t = reset[:, t]
            keep = (~reset_t).to(prev_state.deter.dtype).unsqueeze(-1)
            deter_in = prev_state.deter * keep
            stoch_in = prev_state.stoch * keep.unsqueeze(-1)
            action_t = actions[:, t] * keep

            deter = self._core(deter_in, stoch_in, action_t)
            prior_logits = self._prior(deter)
            stoch, _ = self.sample_stoch(prior_logits)

            valid = valid_mask[:, t].unsqueeze(-1)
            deter = torch.where(valid, deter, prev_state.deter)
            stoch = torch.where(valid.unsqueeze(-1), stoch, prev_state.stoch)
            state = RSSMState(deter=deter, stoch=stoch)

            deter_seq.append(state.deter)
            stoch_seq.append(state.stoch)
            prior_logits_seq.append(prior_logits)

        return {
            "deter": torch.stack(deter_seq, dim=1),
            "stoch": torch.stack(stoch_seq, dim=1),
            "prior_logits": torch.stack(prior_logits_seq, dim=1),
        }


class RSSMWorldModel(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 128, 128),
        action_dim: int = 5,
        deter: int = 1024,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 32,
        blocks: int = 8,
        obs_embed_dim: int = 512,
        cnn_depth: int = 32,
        act: str = "gelu",
        norm: str = "rms",
        unimix: float = 0.01,
        outscale: float = 1.0,
        imglayers: int = 2,
        obslayers: int = 1,
        dynlayers: int = 1,
        absolute: bool = False,
        free_nats: float = 1.0,
        action_embed_dim: int = 128,
        loss_scale_recon: float = 1.0,
        loss_scale_dyn: float = 1.0,
        loss_scale_rep: float = 0.1,
    ):
        super().__init__()
        self.input_shape = cast(Tuple[int, int, int], tuple(input_shape))
        self.free_nats = free_nats
        self.loss_scale_recon = loss_scale_recon
        self.loss_scale_dyn = loss_scale_dyn
        self.loss_scale_rep = loss_scale_rep

        self.encoder = ConvEncoder(self.input_shape, cnn_depth, obs_embed_dim, act)
        self.rssm = DreamerRSSMCore(
            obs_embed_dim=obs_embed_dim,
            action_dim=action_dim,
            deter=deter,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            blocks=blocks,
            act=act,
            norm=norm,
            unimix=unimix,
            outscale=outscale,
            imglayers=imglayers,
            obslayers=obslayers,
            dynlayers=dynlayers,
            absolute=absolute,
            action_embed_dim=0,
        )
        self.decoder = ConvDecoder(
            cast(Tuple[int, int, int], self.input_shape),
            deter_dim=deter,
            stoch=stoch,
            classes=classes,
            cnn_depth=cnn_depth,
            act=act,
            bspace=blocks,
        )

    def _preprocess_video(self, video: Tensor) -> Tensor:
        video = video.float()
        if video.max() > 1.0 or video.min() < 0.0:
            video = video / 255.0
        return video - 0.5

    def decode(self, feat: Dict[str, Tensor]) -> Tensor:
        batch_size, time_steps = feat["deter"].shape[:2]
        return self.decoder(
            feat["deter"],
            feat["stoch"],
            batch_size=batch_size,
            time_steps=time_steps,
        )

    def observe(
        self,
        video: Tensor,
        actions: Tensor,
        reset: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> Tuple[RSSMState, Dict[str, Tensor], Tensor]:
        obs_embed = self.encoder(self._preprocess_video(video))
        state, feat = self.rssm.observe(obs_embed, actions, reset=reset, valid_mask=valid_mask)
        recon = self.decode(feat)
        return state, feat, recon

    def imagine(
        self,
        actions: Tensor,
        initial_state: RSSMState,
        reset: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        feat = self.rssm.imagine(actions, initial_state, reset=reset, valid_mask=valid_mask)
        recon = self.decode(feat)
        return feat, recon

    def _masked_mean(self, loss: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        if valid_mask is None:
            return loss.mean()
        weights = valid_mask.to(loss.dtype)
        denom = weights.sum().clamp_min(1.0)
        return (loss * weights).sum() / denom

    def _categorical_kl(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs = lhs.clamp_min(1e-6)
        rhs = rhs.clamp_min(1e-6)
        return (lhs * (lhs.log() - rhs.log())).sum(dim=-1).sum(dim=-1)

    def training_losses(
        self,
        video: Tensor,
        actions: Tensor,
        reset: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        _, feat, recon = self.observe(video, actions, reset=reset, valid_mask=valid_mask)
        recon_loss = F.mse_loss(recon, video, reduction="none").flatten(start_dim=2).mean(dim=-1)

        prior_probs = self.rssm.probs_from_logits(feat["prior_logits"])
        post_probs = self.rssm.probs_from_logits(feat["post_logits"])
        dyn_loss = self._categorical_kl(post_probs.detach(), prior_probs)
        rep_loss = self._categorical_kl(post_probs, prior_probs.detach())

        if self.free_nats > 0:
            dyn_loss = dyn_loss.clamp_min(self.free_nats)
            rep_loss = rep_loss.clamp_min(self.free_nats)

        recon_scalar = self._masked_mean(recon_loss, valid_mask)
        dyn_scalar = self._masked_mean(dyn_loss, valid_mask)
        rep_scalar = self._masked_mean(rep_loss, valid_mask)
        total = (
            self.loss_scale_recon * recon_scalar
            + self.loss_scale_dyn * dyn_scalar
            + self.loss_scale_rep * rep_scalar
        )

        return {
            "loss": total,
            "recon_loss": recon_scalar,
            "dyn_loss": dyn_scalar,
            "rep_loss": rep_scalar,
            "recon_video": recon,
            "feat": feat,
        }

from omegaconf import DictConfig

class PropertyMixin:

    # ---------------------------------------------------------------------
    # NOTE: n_{frames, tokens} indicates the number of frames/tokens
    # that the model actually processes during training/validation.
    # During validation, it may be different from max_{frames, tokens},
    # ---------------------------------------------------------------------
    @property
    def sampling_timesteps(self) -> int:
        respacing = self.cfg.diffusion.sampling_timesteps
        if respacing == "":
            return self.cfg.diffusion.timesteps
        elif "ddim" in respacing:
            return int(respacing.split("ddim")[1])
        else:
            return int(respacing)

    @property
    def n_context_frames(self) -> int:
        return self.cfg.tasks.prediction.context_frames

    @property
    def n_frames(self) -> int:
        return self.forward_window_size_in_tokens if self.trainer.training else self.cfg.n_frames

    @property
    def n_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_frames)

    @property
    def n_context_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_context_frames)
    
    # ---------------------------------------------------------------------
    # NOTE: max_{frames, tokens} indicates the maximum number of frames/tokens
    # that the model can process within a single forward pass.
    # ---------------------------------------------------------------------
    
    @property
    def forward_window_size_in_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.forward_window_size_in_frames)
    
    @property
    def forward_window_size_in_frames(self):
        return self.backbone_cfg.forward_window_size
    

    def _n_frames_to_n_tokens(self, n_frames: int) -> int:
        """
        Converts the number of frames to the number of tokens.
        - Chunk-wise VideoVAE: 1st frame -> 1st token, then every self.temporal_downsampling_factor frames -> next token.
        - ImageVAE or Non-latent Diffusion: 1 token per frame.
        """
        return (n_frames - 1) // self.temporal_downsampling_factor + 1

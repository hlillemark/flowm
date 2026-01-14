from .data_mixin import DataMixin
from .metric_mixin import MetricMixin
from .optimizer_mixin import OptimizerMixin
from .vae_mixin import VAEMixin
from .checkpoint_mixin import CheckpointMixin
from .logging_mixin import LoggingMixin
from .property_mixin import PropertyMixin

class BaseWMMixin(DataMixin, MetricMixin, OptimizerMixin, VAEMixin, CheckpointMixin, LoggingMixin, PropertyMixin):
    """
    The base mixin for all WM algorithms, so we won't have too verbose code.
    """
    pass
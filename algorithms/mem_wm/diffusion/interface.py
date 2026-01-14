from dataclasses import dataclass
import torch
from typing import Optional
import enum

class CaseInsensitiveEnumMeta(enum.EnumMeta):
    def __getitem__(cls, name: str):
        try:
            return cls._member_map_[name.upper()]
        except KeyError:
            raise ValueError(f"{name} is not a valid {cls.__name__}")

class CaseInsensitiveEnum(enum.Enum, metaclass=CaseInsensitiveEnumMeta):
    pass

class ModelMeanType(CaseInsensitiveEnum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    VELOCITY = enum.auto()  # the model predicts velocity, TODO


class ModelVarType(CaseInsensitiveEnum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(CaseInsensitiveEnum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL



@dataclass
class StepOutput:
    x: torch.Tensor
    pred_xstart: torch.Tensor
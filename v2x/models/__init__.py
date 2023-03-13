import sys
import os
import os.path as osp

from .detection_models import SingleSide, LateFusion, EarlyFusion, VehOnly, InfOnly

SUPPROTED_MODELS = {
    "single_side": SingleSide,
    "late_fusion": LateFusion,
    "early_fusion": EarlyFusion,
    "veh_only": VehOnly,
    "inf_only": InfOnly,
}

from .dair_v2x_for_detection import DAIRV2XI, DAIRV2XV, VICSyncDataset, VICAsyncDataset
from .dataset_utils import *

SUPPROTED_DATASETS = {
    "dair-v2x-v": DAIRV2XV,
    "dair-v2x-i": DAIRV2XI,
    "vic-sync": VICSyncDataset,
    "vic-async": VICAsyncDataset,
}

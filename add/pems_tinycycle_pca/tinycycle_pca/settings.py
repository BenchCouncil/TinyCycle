"""Fixed hyperparameters for the TinyCycle-PCA results in the PEMS table."""

from dataclasses import dataclass

from .model import Config

LOOKBACK = 720
PERIOD = 1
MODEL = Config(period=PERIOD, order=LOOKBACK // PERIOD, center=False, tau=1.0)
HORIZONS = (12, 24, 48, 96)
PCA_RANK = 4
BOUNDARY_FIT_ORIGINS = 512
SPLIT_OFFSET = 96


@dataclass(frozen=True)
class Dataset:
    shape: tuple[int, int]
    md5: str
    base_regularization: float = 0.0
    pca_ridge: float = 0.0


DATASETS = {
    "PEMS03": Dataset((26208, 358), "651add9bb9eaf7f5eda2f2ee8778a182"),
    "PEMS04": Dataset((16992, 307), "5238832691101da85fcac772a32051da", pca_ridge=0.001),
    "PEMS07": Dataset((28224, 883), "978d3d9b85fe640a446983a34271a48d", base_regularization=0.003),
    "PEMS08": Dataset((17856, 170), "2a528d169c0d90294c9e24288a430132"),
}

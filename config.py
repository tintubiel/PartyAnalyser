import typing as tp

from omegaconf import OmegaConf
from pydantic import BaseModel


class DataConfig(BaseModel):
    batch_size: int
    n_workers: int
    train_size: float
    img_width: int
    img_height: int


class Config(BaseModel):
    project_name: str
    experiment_name: str
    data_config: DataConfig
    n_epochs: int
    lr: float
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

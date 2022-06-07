import warnings
from typing import List, Optional
warnings.filterwarnings("ignore")

from comet_ml import Experiment

from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

from pytorch_lightning import Callback, Trainer, LightningDataModule


import hydra
from omegaconf import DictConfig

from models.factory import FACTORY
import utils

log = utils.get_logger(__name__)


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):

    dataset = cfg.datamodule.name
    model = cfg.model.name
    
    print("This run will tune the model", model, "on the", dataset, "dataset")
    seed_everything(cfg.seed, workers=True)

    # Initialize Logger
    comet_logger = CometLogger(
        project_name=f"{model}-tune-{dataset}",
        experiment_name=f"{model}_seed_{cfg.seed}_{dataset}")

    # Initialize the datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=comet_logger, _convert_="partial"
    )

    model = FACTORY[model]
    model = model(cfg.model.params)

    trainer.fit(model, datamodule)
    out = trainer.callback_metrics['val_mae_loss'].item()
    return out

if __name__ == "__main__":

    main()
    
    

    

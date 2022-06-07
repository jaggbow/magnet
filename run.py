import warnings
from typing import List, Optional
warnings.filterwarnings("ignore")

from comet_ml import Experiment

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Callback, Trainer, LightningDataModule

import hydra
from omegaconf import DictConfig

from models.factory import FACTORY
import utils

log = utils.get_logger(__name__)

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):

    dataset = cfg.datamodule.name
    model = cfg.model.name
    
    print("This run trains and tests the model", model, "on the", dataset, "dataset")
    seed_everything(cfg.seed, workers=True)

    # Initialize Logger
    comet_logger = CometLogger(
        project_name=f"pdeone-{dataset.replace('_','-')}",
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

    trainer.fit(model, datamodule) # Train the model
    log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}") # print path to best checkpoint
    # trainer.test(model, datamodule, ckpt_path='best', verbose=True) # Test the model

if __name__ == "__main__":

    main()
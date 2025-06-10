import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base=None, config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    print("Configuration loaded by Hydra:")
    print(cfg)
    
    # 1. Instantiate DataModule from the config
    datamodule = hydra.utils.instantiate(cfg.data)

    # 2. Instantiate Model from the config
    # The _target_ key in the model YAML tells Hydra which class to create.
    model = hydra.utils.instantiate(cfg.model)

    # 3. Instantiate the Trainer and any loggers (like W&B)
    trainer = hydra.utils.instantiate(cfg.trainer)

    # 4. Train the model
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
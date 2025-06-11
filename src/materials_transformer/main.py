import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import pytorch_lightning as pl
from hydra.utils import instantiate
#import pyrootutils

#root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> float:
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")
    
    # -- MLflow --
    mlflow.set_tracking_uri(f"file://{hydra.utils.get_original_cwd()}/mlruns")
    mlflow.set_experiment(cfg.project_name)
    
    with mlflow.start_run() as run:
        # logging hydra params
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        print(f"MLflow Run ID: {run.info.run_id}")
    
        # 1. Instantiate Model and DataModule from the config
        model = hydra.utils.instantiate(cfg.model)
        datamodule = hydra.utils.instantiate(cfg.data)

        # 3. Instantiate the Trainer and any loggers (like W&B)
        trainer = hydra.utils.instantiate(cfg.trainer)

        # 4. Train the model
        trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
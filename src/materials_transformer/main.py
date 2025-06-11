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
    
    '''# -- MLflow --
    mlflow.set_tracking_uri(f"file://{hydra.utils.get_original_cwd()}/mlruns")
    mlflow.set_experiment(cfg.project_name)
    
    with mlflow.start_run() as run:
        # logging hydra params
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        print(f"MLflow Run ID: {run.info.run_id}")'''
    
    # 1. Instantiate associated objects
    model =instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    trainer = instantiate(cfg.trainer)

    # 4. Train the model
    # MLflow should automatically log due to MLflow logger configuration
    trainer.fit(model=model, datamodule=datamodule)
    
    objective_metric_name = cfg.get("objective_metric", "val_loss")
    objective_value = trainer.callback_metrics.get(objective_metric_name)
    
    if objective_value is None:
        print(f"Warning: Objective metric '{objective_metric_name}' not found. Returning infinity.")
        return float("inf")

    # <-- CRUCIAL FOR OPTUNA: Return the final objective value.
    # Hydra's Optuna Sweeper plugin captures this return value and reports it to the Optuna study.
    return float(objective_value)
    
    # where do files get logged?
    # ${paths.X}, need to set that up

if __name__ == "__main__":
    main()
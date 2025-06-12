import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__, 
    indicator=".project-root", 
    pythonpath=True,
    dotenv=True)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path=str(root / "conf"), config_name="config")
def main(cfg: DictConfig) -> float:
    
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")
    
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

    return float(objective_value) # necessary for optuna

if __name__ == "__main__":
    main()
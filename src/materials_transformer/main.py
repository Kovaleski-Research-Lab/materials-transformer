# ---------------------
# Import: Python libs
# ---------------------

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__, 
    indicator=".project-root", 
    pythonpath=True,
    dotenv=True)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import mlflow
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import partial

# ---------------------
# Import: Custom libs
# ---------------------

from utils.eval import create_dft_plot_artifact

@hydra.main(version_base=None, config_path=str(root / "conf"), config_name="config")
def main(cfg: DictConfig) -> float:
    
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")
    
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    
    # 1. Instantiate associated objects
    model =instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    trainer = instantiate(cfg.trainer)
    
    # auto log all MLflow entities
    mlflow.pytorch.autolog(log_models=False)

    # 2. Train the model
    # MLflow should automatically log due to MLflow logger configuration
    trainer.fit(model=model, datamodule=datamodule)
    
    # 3. Prepare evaluation
    run_id = trainer.logger.run_id
    model_uri = f"runs:/{run_id}/model"
    #best_model_path = trainer.checkpoint_callback.best_model_path
    #best_model = type(model).load_from_checkpoint(best_model_path)
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()
    
    # create partial functions for custom artifacts in MLflow
    plot_fn_partial = partial(create_dft_plot_artifact, cfg=cfg, sample_idx=0)
    dft_plot_fn = mlflow.models.make_metric(
        eval_fn=plot_fn_partial,
        greater_is_better=False,
        name="dft_plot_generator",
        long_name="DFT Field Plot Artifact Generator"
    )
    
    # 4. format data and results for evaluation
    eval_data_list = []
    eval_labels_list = []
    for batch in tqdm(test_dataloader, desc="Formatting evalulation data"):
        samples, labels = batch
        samples = samples.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # reshape the data for mlflow
        # OG shape -> [batch, flattened(seq_len, channels, H, W)]
        num_samples = samples.shape[0]
        eval_data_list.append(samples.reshape(num_samples, -1))
        eval_labels_list.append(labels.reshape(num_samples, -1))
        
    # Concatenate all batches into single numpy arrays and make the df
    eval_data = np.concatenate(eval_data_list, axis=0)
    eval_labels = np.concatenate(eval_labels_list, axis=0)
    eval_df = pd.DataFrame(eval_data)
    eval_df["targets"] = eval_labels.tolist()
    
    print(f"Formatted data for evaluation: {eval_df.shape[0]} samples")
    
    
    mlflow.set_tracking_uri(trainer.logger.experiment.tracking_uri) 
    print(f"Running mlflow.evaluate on model: {model_uri}")
    #with mlflow.start_run(run_name="post_training_evaluation") as eval_run:
    #    mlflow.set_tag("mlflow.parent_run_id", run_id)
    #    print(f"Evaluation run id: {eval_run.info.run_id}")
    #    '''mlflow.pytorch.log_model(
    #        pytorch_model=best_model,
    #        name="model"
    #    )'''
    results = mlflow.evaluate(
        model=model_uri,
        data=eval_df,
        targets="targets",
        model_type="regressor",
        evaluators=["default"],
        extra_metrics=[dft_plot_fn]
    )
        
    print("--- mlflow.evaluate() Results ---")
    print(pd.DataFrame(results.metrics, index=["value"]).T)
    
    # --- original objective return for Optuna ---
    objective_metric_name = cfg.get("objective_metric", "val_loss")
    objective_value = trainer.callback_metrics.get(objective_metric_name)
    
    if objective_value is None:
        print(f"Warning: Objective metric '{objective_metric_name}' not found. Returning infinity.")
        return float("inf")

    return float(objective_value)

if __name__ == "__main__":
    main()
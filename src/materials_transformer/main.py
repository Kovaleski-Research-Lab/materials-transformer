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
from mlflow.types.schema import Schema, TensorSpec
from mlflow.models import ModelSignature
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import partial
import tempfile

# ---------------------
# Import: Custom libs
# ---------------------

from utils.eval import create_dft_plot_artifact, CustomVisionEvaluator

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
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = type(model).load_from_checkpoint(best_model_path)
    device = next(best_model.parameters()).device
    
    # get a single batch of the data and infer the signature
    input_sample, output_sample = next(iter(datamodule.train_dataloader()))
    input_sample = input_sample.cpu().numpy()
    
    input_shape = (-1, *input_sample.shape[1:]) # (-1, 1, 2, 166, 166)
    output_shape = (-1, *output_sample.shape[1:])
    input_schema = Schema([TensorSpec(np.dtype(np.float32), input_shape)])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), output_shape)])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    # setup things for testing
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()
    
    # 6. Run inference in batches to avoid memory errors
    print("--- Running manual evaluation loop... ---")
    device = next(best_model.parameters()).device
    best_model.to(device)
    best_model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for samples, labels in tqdm(test_dataloader, desc="Inference on test set"):
            samples = samples.to(device)
            preds = best_model(samples)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batch results
    final_preds = np.concatenate(all_preds, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)

    # 7. Log custom metrics and artifacts to the same run
    mlflow.set_tracking_uri(trainer.logger.experiment.tracking_uri)

    # Calculate a final test MSE and log it
    test_mse = np.mean((final_preds - final_labels) ** 2)
    mlflow.log_metric("test_set_mse", test_mse)
    print(f"Logged test_set_mse: {test_mse:.4f}")

    # Create the plot artifact
    plot_results_dict = {"targets": final_labels, "predictions": final_preds}
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dft_plot_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir, cfg=cfg, sample_idx=0)
        mlflow.log_artifacts(temp_dir, artifact_path="evaluation_plots")
    print("--- Custom plot artifact logged successfully. ---")

    # --- Return objective for Optuna ---
    objective_metric_name = cfg.get("objective_metric", "val_loss")
    objective_value = trainer.callback_metrics.get(objective_metric_name)
    
    if objective_value is None:
        print(f"Warning: Objective metric '{objective_metric_name}' not found. Returning infinity.")
        return float("inf")

    return float(objective_value)

if __name__ == "__main__":
    main()
    
# ARCHIVAL
# old method with model signature

'''# create partial functions for custom artifacts in MLflow
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
    
    # reshape the data to OG for mlflow
    # OG shape -> [batch, flattened(seq_len, channels, H, W)]
    num_samples = samples.shape[0]
    eval_data_list.append(samples.reshape(num_samples, -1))
    eval_labels_list.append(labels.reshape(num_samples, -1))
    break
    
# Concatenate all batches into single numpy arrays and make the df
eval_data = np.concatenate(eval_data_list, axis=0)
eval_labels = np.concatenate(eval_labels_list, axis=0)

eval_df = pd.DataFrame({
    "x": [row for row in eval_data]
})

# 3. Add the 'targets' column as before.
eval_df["targets"] = [row for row in eval_labels]

mlflow.set_tracking_uri(trainer.logger.experiment.tracking_uri) 
model_info = mlflow.pytorch.log_model(
    pytorch_model=best_model,
    name="model",
    signature=signature
)
print(f"Running mlflow.evaluate on model: {model_uri}")
vision_evaluator = CustomVisionEvaluator()
results = vision_evaluator.evaluate(
    model=model_info.model_uri,
    data=eval_df,
    targets= "targets",
    custom_metrics=[dft_plot_fn]
)'''
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from omegaconf import DictConfig
import mlflow
from tqdm import tqdm

import utils.mapping as mapping

def create_dft_plot_artifact(
    eval_df: pd.DataFrame,
    #builtin_metrics: dict,
    artifacts_dir: str,
    cfg:  DictConfig,
    sample_idx: int = 0
) -> dict:
    """
    A custom MLflow metric function for generating and saving DFT field plots.
    """
    print(f"Generating custom DFT plot for sample index {sample_idx}...")
    
    # extract the relevant data
    preds = np.array(eval_df['predictions'][sample_idx])
    truths = np.array(eval_df['targets'][sample_idx])
    
    # clarify configurations
    H = cfg.data.near_field_dim
    W = cfg.data.near_field_dim
    C = 2 # real and imag
    T = cfg.data.seq_len
    
    # use configs to reshape back to [T, C, H, W]
    preds_reshaped = torch.from_numpy(preds).reshape(T, C, H, W)
    truths_reshaped = torch.from_numpy(truths).reshape(T, C, H, W)
    
    # separate real and imaginary components
    pred_real, pred_imag = preds_reshaped[:, 0, :, :], preds_reshaped[:, 1, :, :]
    truth_real, truth_imag = truths_reshaped[:, 0, :, :], truths_reshaped[:, 1, :, :]

    # convert to polar coords
    truth_mag, truth_phase = mapping.cartesian_to_polar(truth_real, truth_imag)
    pred_mag, pred_phase = mapping.cartesian_to_polar(pred_real, pred_imag)
    
    # Create figure WITHOUT creating subplots
    fig = plt.figure(figsize=(4*T + 2, 16))
    
    # Create gridspec with space for labels and column headers
    gs = fig.add_gridspec(5, T + 1,  # 5 rows: header + 4 data rows
                        width_ratios=[0.3] + [1]*T,
                        height_ratios=[0.05] + [1]*4,
                        hspace=0.1,
                        wspace=0.1)
    
    # Create axes for column headers
    header_axs = [fig.add_subplot(gs[0, j]) for j in range(1, T + 1)]
    
    # Create axes for images
    axs = [[fig.add_subplot(gs[i+1, j]) for j in range(1, T + 1)] 
        for i in range(4)]
    
    # Create axes for row labels
    label_axs = [fig.add_subplot(gs[i+1, 0]) for i in range(4)]
    
    title = "DFT Field Progression - Random Test Sample"
    fig.suptitle(title, fontsize=24, y=0.95, fontweight='bold')
    fig.text(0.5, 0.94, "plotplot", ha='center', fontsize=16)
    
    # Add column headers
    for t, ax in enumerate(header_axs):
        ax.axis('off')
        ax.text(0.5, 0.3,
            f't={t+1}',
            ha='center',
            va='center',
            fontsize=20,
            fontweight='bold')
    
    # Add row labels
    row_labels = ['Ground Truth Magnitude',
                'Predicted Magnitude',
                'Ground Truth Phase',
                'Predicted Phase']
    
    for ax, label in zip(label_axs, row_labels):
        ax.axis('off')
        ax.text(0.95, 0.5, 
            label,
            ha='right',
            va='center',
            fontsize=20,
            fontweight='bold')
    
    # Plot sequence
    for t in range(T):
        axs[0][t].imshow(truth_mag[t], cmap='viridis')
        axs[0][t].axis('off')
        
        axs[1][t].imshow(pred_mag[t], cmap='viridis')
        axs[1][t].axis('off')
        
        axs[2][t].imshow(truth_phase[t], cmap='twilight_shifted')
        axs[2][t].axis('off')
        
        axs[3][t].imshow(pred_phase[t], cmap='twilight_shifted')
        axs[3][t].axis('off')
        
    # save the artifact
    plot_path = os.path.join(artifacts_dir, f"dft_comparison_idx{sample_idx}.pdf")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved plot artifact to {plot_path}")
    
    return {}

# ---------------------------
# Custom MLflow Evaluator
# ---------------------------

class CustomVisionEvaluator(mlflow.models.evaluation.ModelEvaluator):
    def can_evaluate(self, model_type, **kwargs):
        # This evaluator can handle any model type since we define the logic.
        return True

    def evaluate(self, model, data, targets, custom_metrics, artifacts_dir, **kwargs):
        # Create a PyTorch DataLoader to handle batching automatically
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=kwargs.get("batch_size", 8))
        
        device = next(model.parameters()).device
        
        all_preds = []
        with torch.no_grad():
            for (batch_data,) in tqdm(loader, desc="Custom Evaluator Prediction"):
                preds = model(batch_data.to(device))
                all_preds.append(preds.cpu().numpy())
        
        predictions = np.concatenate(all_preds, axis=0)

        # The 'data' is our input, 'targets' is ground truth. Create the eval_df.
        # This is what our custom plotting function expects.
        eval_df = pd.DataFrame({"prediction": [p for p in predictions]})
        
        # Call all the custom metric/artifact functions provided
        for metric_fn in custom_metrics:
            metric_fn(eval_df=eval_df, targets=targets, artifacts_dir=artifacts_dir)
            
        return {"metrics": {}}
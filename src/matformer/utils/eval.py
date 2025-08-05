import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
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
    sample_idx: int = 0
) -> None:
    """
    A custom MLflow metric function for generating and saving DFT field plots.
    """
    print(f"Generating custom DFT plot for sample index {sample_idx}...")
    
    # extract the data for the specified sample
    preds = eval_df['predictions'][sample_idx]
    targets = eval_df['target'][sample_idx]
    
    # clarify configurations
   # H = cfg.data.near_field_dim
    #W = cfg.data.near_field_dim
    #C = 2 # real and imag
    #T = cfg.data.seq_len
    T = preds.shape[0]
    
    # convert to torch tensors and ensure float32
    preds_tensor = torch.from_numpy(preds).float()
    target_tensor = torch.from_numpy(targets).float() 
    
    # separate real and imaginary components
    preds_real, preds_imag = preds_tensor[:, 0, :, :], preds_tensor[:, 1, :, :]
    target_real, target_imag = target_tensor[:, 0, :, :], target_tensor[:, 1, :, :]

    # convert to polar coords
    truth_mag, truth_phase = mapping.cartesian_to_polar(target_real, target_imag)
    pred_mag, pred_phase = mapping.cartesian_to_polar(preds_real, preds_imag)
    
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
    print(f"Saved dft plot artifact to {plot_path}")
    
    return None

def create_correlation_plot_artifact(
    eval_df: pd.DataFrame, 
    artifacts_dir: str,
) -> None:
    """
    Create scatter plots comparing ground truth vs predicted values to visualize correlation,
    averaged across all test samples
    """
    # extract the data for the specified sample
    preds = eval_df['predictions'] # [N, T, C, H, W]
    targets = eval_df['target'] # [N, T, C, H, W]
    n_samples = preds.shape[0]
    
    # separate real and imaginary components for the final slice and flatten - # [N * H * W]
    preds_real_flat, preds_imag_flat = preds[:, -1, 0, :, :].flatten(), preds[:, -1, 1, :, :].flatten()
    targets_real_flat, targets_imag_flat = targets[:, -1, 0, :, :].flatten(), targets[:, -1, 1, :, :].flatten()
    
    # Compute correlations
    corr_real = np.corrcoef(targets_real_flat, preds_real_flat)[0, 1]
    corr_imag = np.corrcoef(targets_imag_flat, preds_imag_flat)[0, 1]
        
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 8))
    
    # 1. Combined density scatter plot
    ax1 = plt.subplot(121)
    
    # Plot real and imaginary components with different colors
    hist2d_real = plt.hist2d(targets_real_flat, preds_real_flat, bins=100,
                            cmap='Reds', norm=matplotlib.colors.LogNorm(),
                            alpha=0.6)
    hist2d_imag = plt.hist2d(targets_imag_flat, preds_imag_flat, bins=100,
                            cmap='Blues', norm=matplotlib.colors.LogNorm(),
                            alpha=0.6)
    
    # Add diagonal line
    min_val = min(targets_real_flat.min(), targets_imag_flat.min(),
                    preds_real_flat.min(), preds_imag_flat.min())
    max_val = max(targets_real_flat.max(), targets_imag_flat.max(),
                    preds_real_flat.max(), preds_imag_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Ground Truth Field Value')
    plt.ylabel('Predicted Field Value')
    plt.title(f'Default Split Testing\nReal (r={corr_real:.4f}) & Imaginary (r={corr_imag:.4f})\nAveraged across {n_samples} samples')
    
    # Custom legend
    legend_elements = [Patch(facecolor='red', alpha=0.6, label='Real Component'),
        Patch(facecolor='blue', alpha=0.6, label='Imaginary Component'),
        plt.Line2D([0], [0], color='k', linestyle='--', label='Perfect Correlation')
    ]
    plt.legend(handles=legend_elements)
    plt.axis('square')
    
    # 2. Combined distribution plot
    ax2 = plt.subplot(122)
    
    # Create histograms with both density and counts
    counts_real, bins, _ = plt.hist(targets_real_flat, bins=100, alpha=0.3, 
                                color='red', label='Ground Truth (Real)', density=True)
    plt.hist(preds_real_flat, bins=bins, alpha=0.3, 
            color='darkred', label='Prediction (Real)', density=True)
    
    counts_imag, bins, _ = plt.hist(targets_imag_flat, bins=100, alpha=0.3, 
                                color='blue', label='Ground Truth (Imag)', density=True)
    plt.hist(preds_imag_flat, bins=bins, alpha=0.3, 
            color='darkblue', label='Prediction (Imag)', density=True)
    
    plt.xlabel('Field Value')
    plt.ylabel('Density')
    
    # Add second y-axis with average counts per sample
    ax2_counts = ax2.twinx()
    bin_width = bins[1] - bins[0]
    max_count = max(
        max(counts_real) * len(targets_real_flat) * bin_width / n_samples,
        max(counts_imag) * len(targets_imag_flat) * bin_width / n_samples
    )
    ax2_counts.set_ylim(0, max_count)
    ax2_counts.set_ylabel('Average Pixel Count per Sample')
    
    plt.title(f'Distribution of Field Values\nReal & Imaginary Components\nAveraged across {n_samples} samples')
    ax2.legend()
        
    # Calculate statistics
    stats_text = (
        'Real Component:\n'
        f'    Mean (Truth/Pred): {targets_real_flat.mean():.3f}/{preds_real_flat.mean():.3f}\n'
        f'    Std (Truth/Pred):  {targets_real_flat.std():.3f}/{preds_real_flat.std():.3f}\n'
        f'    MAE: {np.mean(np.abs(targets_real_flat - preds_real_flat)):.3f}\n'
        f'    RMSE: {np.sqrt(np.mean((targets_real_flat - preds_real_flat)**2)):.3f}\n'
        '\nImaginary Component:\n'
        f'    Mean (Truth/Pred): {targets_imag_flat.mean():.3f}/{preds_imag_flat.mean():.3f}\n'
        f'    Std (Truth/Pred):  {targets_imag_flat.std():.3f}/{preds_imag_flat.std():.3f}\n'
        f'    MAE: {np.mean(np.abs(targets_imag_flat - preds_imag_flat)):.3f}\n'
        f'    RMSE: {np.sqrt(np.mean((targets_imag_flat - preds_imag_flat)**2)):.3f}'
    )
    
    plt.text(0.05, 0.95, stats_text, 
            transform=ax2.transAxes,
            verticalalignment='top', 
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=1),
            fontsize=10)
    
    plt.tight_layout()

    # save the artifact
    plot_path = os.path.join(artifacts_dir, "correlation_analysis.pdf")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved correlation plot artifact to {plot_path}")
    
    return None

def create_flipbook_artifact(
    eval_df: pd.DataFrame,
    artifacts_dir: str,
    sample_idx: int = 0
) -> None:
    """
    Create an animation of 2D field slices across the propagation volume.
    """
    # extract the data for the specified sample
    preds = eval_df['predictions'][sample_idx]
    targets = eval_df['target'][sample_idx]
    frames = preds.shape[0] # number of slices
    
    # convert to torch tensors and ensure float32
    preds_tensor = torch.from_numpy(preds).float()
    target_tensor = torch.from_numpy(targets).float() 
    
    # separate real and imaginary components and reshape for animation
    preds_real, preds_imag = preds_tensor[:, 0, :, :], preds_tensor[:, 1, :, :]
    target_real, target_imag = target_tensor[:, 0, :, :], target_tensor[:, 1, :, :]
    target_real = target_real.permute(1, 2, 0)
    target_imag = target_imag.permute(1, 2, 0)
    preds_real = preds_real.permute(1, 2, 0)
    preds_imag = preds_imag.permute(1, 2, 0)

    # convert to polar coords
    target_mag, target_phase = mapping.cartesian_to_polar(target_real, target_imag)
    pred_mag, pred_phase = mapping.cartesian_to_polar(preds_real, preds_imag)
    
    # bundle
    fields_list = [target_mag, pred_mag, target_phase, pred_phase]
    identifier_list = ['True Magnitude', 'Predicted Magnitude', 'True Phase', 'Predicted Phase']
    
    # generate the animations
    anim_dir = os.path.join(artifacts_dir, "flipbooks")
    os.makedirs(anim_dir, exist_ok=True)
    for fields, identifier in zip(fields_list, identifier_list):
        # like to have distinct color maps for magnitude vs phase
        cmap = 'viridis' if 'Magnitude' in identifier else 'twilight_shifted'
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Initialize: Frame 0
        im = ax.imshow(fields[:, :, 0], cmap=cmap, animated=True)
        ax.set_title(f'{identifier} - Frame 0/{frames}')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # frame updating
        def update(frame):
            im.set_array(fields[:, :, frame])
            ax.set_title(f'{identifier} - Frame {frame}/{frames}')
            return [im]
        
        # Create: Animation object
        anim = FuncAnimation(
            fig, 
            update,
            frames=frames,
            interval=250, # ms between frames 
            blit=True
        )
        
        # save the artifact
        anim_path = os.path.join(anim_dir, f"{identifier} {sample_idx}.gif")
        anim.save(anim_path)
        plt.close(fig)
        print(f"Saved a field animation artifact to {anim_path}")
        
# IR Project plotting
        
def create_matrix_artifact(
    eval_df: pd.DataFrame,
    artifacts_dir: str,
    threshold: float = 0.5
) -> None:
    """Plot the first three actual vs. predicted matrices."""
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    
    actuals = eval_df['target']
    logits = torch.tensor(eval_df['predictions'])
    preds_binary = (torch.sigmoid(logits) > threshold).float()

    for i in range(3):
        axes[i, 0].imshow(actuals[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Actual Matrix {i + 1}')

        axes[i, 1].imshow(preds_binary[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Predicted Matrix {i + 1}')

    plt.tight_layout()
    
    # save the artifact
    plot_path = os.path.join(artifacts_dir, "matrix_visualizations.pdf")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved correlation plot artifact to {plot_path}")
    
    return None

def create_matrix_artifact(
    eval_df: pd.DataFrame,
    artifacts_dir: str
) -> None:
    pass
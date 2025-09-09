import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import hann, blackman
import argparse

#--------------------------------
# Core loss function logic
#--------------------------------

class K_losses:
    def __init__(self, truth, pred, num_bins):
        """Various loss functions and metrics

        Parameters
        ----------
        truth (torch.complex): field ground truths [xdim,ydim]
        pred (torch.complex): model predictions [xdim,ydim]
        num_bins: the number of bins for radial and angular functions
        """
        self.gt = truth
        self.pred = pred
        self.device = truth.device
        self.num_bins = num_bins
        self.gt_k = torch.fft.fft2(self.gt, norm="ortho")
        self.pred_k = torch.fft.fft2(self.pred, norm="ortho")
        n, h, w = self.gt_k.shape
        # compute these just once
        self.polar_coords = precompute_polar_coordinates(h, w, self.device)
        self.radial_bin_edges = torch.linspace(0, torch.hypot(torch.tensor(h/2), torch.tensor(w/2)).item(), num_bins + 1, device=self.device)
        self.angle_bin_edges = torch.linspace(0, 360, num_bins + 1, device=self.device)
        
    def mse(self):
        return torch.mean(torch.abs(self.gt - self.pred)**2)
    
    def kMag(self, option="direct"):
        """k-space magnitude loss. Calculates MSE between magnitudes of k-space ground truths
        and predictions.
        
        Parameters
        ----------
        option (str): calculation method. 'log' if using log transform, 'direct' otherwise.
        """
        mag_gt_k = torch.abs(self.gt_k)
        mag_pred_k = torch.abs(self.pred_k)
        if option == 'direct': # direct
            return torch.mean((mag_gt_k - mag_pred_k)**2)
        elif option == "log":
            return torch.mean((torch.log1p(mag_gt_k) - torch.log1p(mag_pred_k))**2)
        else:
            raise ValueError(f"option: {option} is not recognized.")
    
    def kPhase(self, option="direct"):
        """k-space phase loss. Calculates MSE of wrapped phase difference between ground truths
        and predictions.
        
        Parameters
        ----------
        option (str): calculation method. 'mag_weight' if weighting by magnitudes, 'direct' otherwise.
        """
        phase_gt_k = torch.angle(self.gt_k)
        phase_pred_k = torch.angle(self.pred_k)
        phase_diff = phase_gt_k - phase_pred_k
        wrapped_phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        if option == "direct":
            return torch.mean(wrapped_phase_diff**2)
        elif option == 'mag_weight': # direct
            weights = torch.log1p(torch.abs(self.gt_k) * torch.abs(self.pred_k))
            weights = weights / (torch.max(weights) + 1e-8)
            return torch.mean(weights * wrapped_phase_diff**2)
        else:
            raise ValueError(f"option: {option} is not recognized.")
    
    def kRadial(self):
        """k-space radial loss. Returns the MSE of radial profiles (annular binning) between ground
        truths and predictions.
        
        Parameters
        ----------
        num_bins (int): number of bins for the radial profiling.
        """
        gt_k_radial = get_radial_profile(self.gt_k, self.polar_coords[0], self.radial_bin_edges, self.num_bins)
        pred_k_radial = get_radial_profile(self.pred_k, self.polar_coords[0], self.radial_bin_edges, self.num_bins)
        return torch.mean((gt_k_radial.float().to(self.device) - 
                           pred_k_radial.float().to(self.device))**2)
    
    def kAngular(self):
        """k-space angular loss. Returns the MSE of angular profiles between ground truths and predictions.
        """
        gt_k_angular = get_angular_profile(self.gt_k, self.polar_coords[1], self.angle_bin_edges, self.num_bins)
        pred_k_angular = get_angular_profile(self.pred_k, self.polar_coords[1], self.angle_bin_edges, self.num_bins)
        return torch.mean((gt_k_angular.float().to(self.device) - 
                           pred_k_angular.float().to(self.device))**2)
    
#--------------------------------
# HELPER FUNCTIONS for profile calculation
#--------------------------------

def precompute_polar_coordinates(H, W, device):
    y = torch.arange(0, H, device=device, dtype=torch.float32)
    x = torch.arange(0, W, device=device, dtype=torch.float32)
    y_indices, x_indices = torch.meshgrid(y, x, indexing='ij')
    crow, ccol = H // 2, W // 2
    x_coords = x_indices - ccol
    y_coords = y_indices - crow
    rho = torch.sqrt(x_coords**2 + y_coords**2) # (H, W)
    theta = (torch.arctan2(y_coords, x_coords) * 180 / torch.pi) % 360 # (H, W)
    return rho, theta
    
def get_radial_profile(field, rho, radial_bin_edges, num_bins):
    """Computes 1D radial profiles (annular binning) of an input field.
    
    Parameters
    ----------
    field (torch.tensor): Input field of shape [xdim, ydim]
    rho (torch.tensor): Polar rho component
    radial_bin_edges (torch.tensor): binning setup
    num_bins (int): number of bins for the radial profiling.
    """
    N, H, W = field.shape
    
    # 1. Get spatial amplitude map
    magnitude_maps = torch.abs(field)
    
    # Ensure DC is at center for coordinate calculations:
    mag_maps_shifted = torch.fft.fftshift(magnitude_maps, dim=(-2, -1))
    
    # expand rho for broadcasting
    rho_expanded = rho.unsqueeze(0)
    
    radial_hist = torch.zeros(N, num_bins, device=field.device)    
    for i in range(num_bins):
        mask = (rho_expanded >= radial_bin_edges[i]) & (rho_expanded < radial_bin_edges[i+1])
        # summing over (H,W) dims for each item in batch
        radial_hist[:, i] = torch.sum(mag_maps_shifted * mask, dim=(-2,-1))

    # Normalize
    epsilon = 1e-10
    radial_hist_norm = radial_hist / (torch.sum(radial_hist, dim=1, keepdim=True) + epsilon)
    return radial_hist_norm

def get_angular_profile(field, theta, angle_bin_edges, num_bins):
    """Computes 1D angular profiles of an input field.
    
    Parameters
    ----------
    field (torch.tensor): Input field of shape [xdim, ydim]
    theta (torch.tensor): Polar theta component
    angle_bin_edges (torch.tensor): binning setup
    num_bins (int): number of bins for the angular profiling.
    """
    N, H, W = field.shape
    
    # 1. Get spatial amplitude map
    magnitude_maps = torch.abs(field)
    
    # Ensure DC is at center for coordinate calculations:
    mag_maps_shifted = torch.fft.fftshift(magnitude_maps, dim=(-2, -1))
    
    theta_expanded = theta.unsqueeze(0)
    
    angle_hist = torch.zeros(N, num_bins, device=field.device)
    for i in range(num_bins):
        if i == num_bins - 1:
            mask_angle = (theta_expanded >= angle_bin_edges[i]) & (theta_expanded <= angle_bin_edges[i+1])
        else:
            mask_angle = (theta_expanded >= angle_bin_edges[i]) & (theta_expanded < angle_bin_edges[i+1])
        angle_hist[:, i] = torch.sum(mag_maps_shifted * mask_angle, dim=(-1, -2))
    
    # Normalize
    epsilon = 1e-10
    angle_hist_norm = angle_hist / (torch.sum(angle_hist, dim=1, keepdim=True) + epsilon)
    return angle_hist_norm

#--------------------------------
# HELPER FUCNTIONS - plotting
#--------------------------------

def plot_spatial_comparison(ax_gt, ax_pred, ax_diff, gt_spatial_np, pred_spatial_np, mse_val, sample_idx):
    # Plot Ground Truth Spatial Magnitude
    im_gt = ax_gt.imshow(np.abs(gt_spatial_np), cmap='viridis')
    ax_gt.set_title(f'GT Spatial Mag (Sample {sample_idx})')
    ax_gt.set_xlabel('X'); ax_gt.set_ylabel('Y')
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    # Plot Prediction Spatial Magnitude
    im_pred = ax_pred.imshow(np.abs(pred_spatial_np), cmap='viridis', 
                             vmin=np.min(np.abs(gt_spatial_np)), vmax=np.max(np.abs(gt_spatial_np))) # Use GT scale
    ax_pred.set_title(f'Pred Spatial Mag (MSE: {mse_val:.3e})')
    ax_pred.set_xlabel('X'); ax_pred.set_ylabel('Y')
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

    # Plot Difference
    diff_map = np.abs(np.abs(gt_spatial_np) - np.abs(pred_spatial_np))
    im_diff = ax_diff.imshow(diff_map, cmap='magma')
    ax_diff.set_title(f'Abs Diff Spatial Mag')
    ax_diff.set_xlabel('X'); ax_diff.set_ylabel('Y')
    plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)


def plot_kspace_magnitude_comparison(ax_gt, ax_pred, ax_diff, gt_k_np, pred_k_np, kmag_val, sample_idx):
    gt_k_mag_log_shifted = np.log1p(np.abs(np.fft.fftshift(gt_k_np)))
    pred_k_mag_log_shifted = np.log1p(np.abs(np.fft.fftshift(pred_k_np)))

    im_gt = ax_gt.imshow(gt_k_mag_log_shifted, cmap='afmhot')
    ax_gt.set_title(f'GT k-Mag Log (Sample {sample_idx})')
    ax_gt.set_xlabel('$k_x$'); ax_gt.set_ylabel('$k_y$')
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    im_pred = ax_pred.imshow(pred_k_mag_log_shifted, cmap='afmhot',
                             vmin=np.min(gt_k_mag_log_shifted), vmax=np.max(gt_k_mag_log_shifted)) # Use GT scale
    ax_pred.set_title(f'Pred k-Mag Log (kMag: {kmag_val:.3e})')
    ax_pred.set_xlabel('$k_x$'); ax_pred.set_ylabel('$k_y$')
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
    
    diff_map_k = np.abs(gt_k_mag_log_shifted - pred_k_mag_log_shifted)
    im_diff = ax_diff.imshow(diff_map_k, cmap='magma')
    ax_diff.set_title(f'Abs Diff k-Mag Log')
    ax_diff.set_xlabel('$k_x$'); ax_diff.set_ylabel('$k_y$')
    plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)


def plot_kspace_phase_comparison(ax_gt, ax_pred, ax_diff, gt_k_np, pred_k_np, kphase_val, sample_idx):
    gt_k_phase_shifted = np.angle(np.fft.fftshift(gt_k_np))
    pred_k_phase_shifted = np.angle(np.fft.fftshift(pred_k_np))
    
    phase_diff_raw = gt_k_phase_shifted - pred_k_phase_shifted
    wrapped_phase_diff_map = np.arctan2(np.sin(phase_diff_raw), np.cos(phase_diff_raw))

    im_gt = ax_gt.imshow(gt_k_phase_shifted, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax_gt.set_title(f'GT k-Phase (Sample {sample_idx})')
    ax_gt.set_xlabel('$k_x$'); ax_gt.set_ylabel('$k_y$')
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi])

    im_pred = ax_pred.imshow(pred_k_phase_shifted, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax_pred.set_title(f'Pred k-Phase (kPhase: {kphase_val:.3e})')
    ax_pred.set_xlabel('$k_x$'); ax_pred.set_ylabel('$k_y$')
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi])
    
    im_diff = ax_diff.imshow(np.abs(wrapped_phase_diff_map), cmap='viridis', vmin=0, vmax=np.pi) # Plot magnitude of wrapped diff
    ax_diff.set_title(f'Abs Wrapped Phase Diff')
    ax_diff.set_xlabel('$k_x$'); ax_diff.set_ylabel('$k_y$')
    plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04, ticks=[0, np.pi/2, np.pi])


def plot_kspace_radial_profiles(ax, gt_profile_np, pred_profile_np, kradial_val, bin_edges_np, num_bins, sample_idx):
    bin_centers = 0.5 * (bin_edges_np[:-1] + bin_edges_np[1:])
    if len(bin_centers) != len(gt_profile_np): # Fallback for bin_centers calculation
        bin_centers = np.arange(len(gt_profile_np))

    ax.plot(bin_centers, gt_profile_np, label='GT k-Radial', color='blue', linestyle='-', marker='o', markersize=3, alpha=0.8)
    ax.plot(bin_centers, pred_profile_np, label=f'Pred k-Radial (Loss: {kradial_val:.3e})', color='red', linestyle='--', marker='x', markersize=3, alpha=0.8)
    ax.set_xlabel('Spatial Frequency Magnitude ($k_r$)')
    ax.set_ylabel('Norm. Sum of Magnitudes')
    ax.set_title(f'k-space Radial Profile (S {sample_idx})')
    ax.legend(fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_kspace_angular_profiles_polar(fig, position_spec, gt_profile_np, pred_profile_np, kangular_val, num_bins, sample_idx):
    # Special handling for polar plot as it needs its own projection
    ax = fig.add_subplot(position_spec, projection='polar')
    
    bin_width_deg = 360.0 / num_bins
    angles_deg_centers = np.linspace(0, 360 - bin_width_deg, num_bins) + bin_width_deg / 2.0
    angles_rad_centers = np.deg2rad(angles_deg_centers)

    plot_angles_rad = np.append(angles_rad_centers, angles_rad_centers[0])
    plot_gt_profile = np.append(gt_profile_np, gt_profile_np[0])
    plot_pred_profile = np.append(pred_profile_np, pred_profile_np[0])

    ax.plot(plot_angles_rad, plot_gt_profile, label='GT k-Angular', color='blue', linestyle='-', marker='o', markersize=3)
    ax.plot(plot_angles_rad, plot_pred_profile, label=f'Pred k-Angular (Loss: {kangular_val:.3e})', color='red', linestyle='--', marker='x', markersize=3)
    
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    max_r_value = max(np.max(plot_gt_profile), np.max(plot_pred_profile), 0.01) # ensure not zero
    ax.set_rlim(0, max_r_value * 1.15)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_title(f'k-space Angular Profile (S {sample_idx})', y=1.12) # Adjust y for title position
    ax.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1.45, 1.15))
    ax.grid(True, linestyle='--', alpha=0.6)

#--------------------------------
# Main Diagnostic Workflow
#--------------------------------

def run_diagnostics(data_path, num_samples_to_analyze=3, num_radial_bins=32, num_angular_bins=36):
    try:
        data_content = torch.load(data_path)
        # Assuming data_content is the dictionary directly
        nf_truth_all = data_content['nf_truth']
        nf_pred_all = data_content['nf_pred']
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except KeyError as e:
        print(f"Error: Key {e} not found in the data file. Ensure 'nf_truth' and 'nf_pred' exist.")
        return

    num_available_samples = nf_truth_all.shape[0]
    if num_available_samples == 0:
        print("No samples found in the data file.")
        return
        
    # Ensure data is float before converting to complex
    # Data seems to be stored as [sample, real/imag, H, W]
    # This also assumes data is stored as numpy arrays and needs conversion to torch tensors.
    # If it's already torch tensors, adjust accordingly.
    
    # Determine device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_metrics_list = []

    for sample_idx in range(min(num_samples_to_analyze, num_available_samples)):
        print(f"\n--- Processing Sample {sample_idx} ---")
        
        # Assuming nf_truth_all and nf_pred_all are NumPy arrays from torch.load if .pt contained NumPy
        # Ensure they are float before making complex
        if len(nf_truth_all.shape) == 5: # preds are volumes
            gt_spatial_real = torch.from_numpy(nf_truth_all[sample_idx, 0, 0, :, :].astype(np.float32)).to(device)
            gt_spatial_imag = torch.from_numpy(nf_truth_all[sample_idx, 0, 1, :, :].astype(np.float32)).to(device)
            pred_spatial_real = torch.from_numpy(nf_pred_all[sample_idx, 0, 0, :, :].astype(np.float32)).to(device)
            pred_spatial_imag = torch.from_numpy(nf_pred_all[sample_idx, 0, 1, :, :].astype(np.float32)).to(device)
        else:
            gt_spatial_real = torch.from_numpy(nf_truth_all[sample_idx, 0, :, :].astype(np.float32)).to(device)
            gt_spatial_imag = torch.from_numpy(nf_truth_all[sample_idx, 1, :, :].astype(np.float32)).to(device)
            pred_spatial_real = torch.from_numpy(nf_pred_all[sample_idx, 0, :, :].astype(np.float32)).to(device)
            pred_spatial_imag = torch.from_numpy(nf_pred_all[sample_idx, 1, :, :].astype(np.float32)).to(device)

        gt_complex = torch.complex(gt_spatial_real, gt_spatial_imag)
        pred_complex = torch.complex(pred_spatial_real, pred_spatial_imag)
        
        losses_obj = K_losses(gt_complex, pred_complex)
        
        metrics = {
            'sample_idx': sample_idx,
            'mse': losses_obj.mse().item(),
            'kmag_log': losses_obj.kMag(option="log").item(),
            'kphase_mw': losses_obj.kPhase(option="mag_weight").item(), # Using magnitude weighted
            'kradial': losses_obj.kRadial(num_bins=num_radial_bins).item(),
            'kangular': losses_obj.kAngular(num_bins=num_angular_bins).item()
        }
        all_metrics_list.append(metrics)

        print(f"Calculated Metrics for sample {sample_idx}:")
        for key, val in metrics.items():
            if key != 'sample_idx':
                print(f"  {key}: {val:.6e}")
        
        # --- Generate Multi-Panel Plot for this sample ---
        fig = plt.figure(figsize=(22, 12)) # Increased figure size
        gs = fig.add_gridspec(2, 3, width_ratios=[1,1,1], height_ratios=[1,1]) # GridSpec for flexible layout

        # Panel 1: Spatial Domain (3 subplots for GT, Pred, Diff)
        ax_sp_gt = fig.add_subplot(gs[0, 0])
        ax_sp_pred = fig.add_subplot(gs[0, 1]) #, sharex=ax_sp_gt, sharey=ax_sp_gt) # Removed share for individual colorbars
        ax_sp_diff = fig.add_subplot(gs[0, 2]) #, sharex=ax_sp_gt, sharey=ax_sp_gt)
        plot_spatial_comparison(ax_sp_gt, ax_sp_pred, ax_sp_diff,
                                gt_complex.cpu().numpy(), pred_complex.cpu().numpy(), 
                                metrics['mse'], sample_idx)

        # Panel 2: k-space Magnitude (3 subplots for GT, Pred, Diff)
        # Need to re-create these axes as they will be used by new set of 3 plots
        fig_kmag = plt.figure(figsize=(22,5)) # Separate figure for k-mag for clarity or use nested gridspec
        gs_kmag = fig_kmag.add_gridspec(1,3)
        ax_km_gt = fig_kmag.add_subplot(gs_kmag[0,0])
        ax_km_pred = fig_kmag.add_subplot(gs_kmag[0,1])
        ax_km_diff = fig_kmag.add_subplot(gs_kmag[0,2])
        plot_kspace_magnitude_comparison(ax_km_gt, ax_km_pred, ax_km_diff,
                                       losses_obj.gt_k.cpu().numpy(), losses_obj.pred_k.cpu().numpy(),
                                       metrics['kmag_log'], sample_idx)
        fig_kmag.suptitle(f"K-Space Magnitude Comparison (Sample {sample_idx})", fontsize=14)
        fig_kmag.tight_layout(rect=[0,0,1,0.96])
        fig_kmag.show()


        # Panel 3: k-space Phase (3 subplots for GT, Pred, Diff)
        fig_kphase = plt.figure(figsize=(22,5))
        gs_kphase = fig_kphase.add_gridspec(1,3)
        ax_kph_gt = fig_kphase.add_subplot(gs_kphase[0,0])
        ax_kph_pred = fig_kphase.add_subplot(gs_kphase[0,1])
        ax_kph_diff = fig_kphase.add_subplot(gs_kphase[0,2])
        plot_kspace_phase_comparison(ax_kph_gt, ax_kph_pred, ax_kph_diff,
                                        losses_obj.gt_k.cpu().numpy(), losses_obj.pred_k.cpu().numpy(),
                                        metrics['kphase_mw'], sample_idx)
        fig_kphase.suptitle(f"K-Space Phase Comparison (Sample {sample_idx})", fontsize=14)
        fig_kphase.tight_layout(rect=[0,0,1,0.96])
        fig_kphase.show()


        # Panel 4 & 5: Radial and Angular k-space Profiles (Sharing a row in main fig)
        ax_krad = fig.add_subplot(gs[1, 0])
        # For polar plot, it's better to create it with projection='polar' directly
        # We will create the polar axis using the main figure object and gridspec
        # axs[1,1] was implicitly created by gs. We replace it.
        
        # Calculate profiles for plotting
        gt_k_radial_profile_np = get_radial_profile(losses_obj.gt_k.cpu().numpy(), num_radial_bins)
        pred_k_radial_profile_np = get_radial_profile(losses_obj.pred_k.cpu().numpy(), num_radial_bins)
        h_k, w_k = losses_obj.gt_k.shape # Use shape of k-space tensor
        max_radius_k = np.sqrt((h_k / 2)**2 + (w_k / 2)**2)
        radial_bin_edges_k = np.linspace(0, max_radius_k, num_radial_bins + 1)
        
        plot_kspace_radial_profiles(ax_krad, gt_k_radial_profile_np, pred_k_radial_profile_np,
                                    metrics['kradial'], radial_bin_edges_k, num_radial_bins, sample_idx)

        # Polar plot for angular profile
        gt_k_angular_profile_np = get_angular_profile(losses_obj.gt_k.cpu().numpy(), num_angular_bins)
        pred_k_angular_profile_np = get_angular_profile(losses_obj.pred_k.cpu().numpy(), num_angular_bins)
        
        # Use the new dedicated polar plotting function that takes fig and gridspec position
        plot_kspace_angular_profiles_polar(fig, gs[1,1], gt_k_angular_profile_np, pred_k_angular_profile_np,
                                           metrics['kangular'], num_angular_bins, sample_idx)
        
        # Optional: Text summary in the last panel
        ax_summary = fig.add_subplot(gs[1, 2])
        ax_summary.axis('off')
        loss_summary_text = f"Sample: {sample_idx}\n" + "\n".join([f"{key}: {val:.3e}" for key, val in metrics.items() if key != 'sample_idx'])
        ax_summary.text(0.05, 0.95, loss_summary_text, transform=ax_summary.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

        fig.suptitle(f"Comprehensive Diagnostics for Sample {sample_idx}", fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle
        fig.show() # Show the main figure with spatial, radial, angular, summary

    # --- Optional: Summary Scatter Plots after loop ---
    if all_metrics_list:
        try:
            import pandas as pd
            df_metrics = pd.DataFrame(all_metrics_list)
            
            plt.figure(figsize=(8,6))
            plt.scatter(df_metrics['mse'], df_metrics['kangular'], alpha=0.7)
            plt.xlabel("Spatial MSE")
            plt.ylabel("k-space Angular Profile MSE")
            plt.title("Spatial MSE vs. k-Angular Profile MSE")
            plt.grid(True, linestyle='--', alpha=0.6)
            for i, row in df_metrics.iterrows():
                plt.annotate(str(row['sample_idx']), (row['mse'], row['kangular']))
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8,6))
            plt.scatter(df_metrics['mse'], df_metrics['kradial'], alpha=0.7)
            plt.xlabel("Spatial MSE")
            plt.ylabel("k-space Radial Profile MSE")
            plt.title("Spatial MSE vs. k-Radial Profile MSE")
            plt.grid(True, linestyle='--', alpha=0.6)
            for i, row in df_metrics.iterrows():
                plt.annotate(str(row['sample_idx']), (row['mse'], row['kradial']))
            plt.tight_layout()
            plt.show()
            
            # Add more scatter plots as needed (e.g., mse vs kmag_log)

        except ImportError:
            print("\nPandas not installed. Skipping summary scatter plots.")
        except Exception as e:
            print(f"Error generating summary plots: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diagnostic analysis on E-field predictions.")
    parser.add_argument("--data_path", type=str, help="Path to the .pt data file (e.g., sim.pt)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to analyze and plot.")
    parser.add_argument("--radial_bins", type=int, default=100, help="Number of bins for radial profiles.")
    parser.add_argument("--angular_bins", type=int, default=100, help="Number of bins for angular profiles.")
    
    args = parser.parse_args()
    
    run_diagnostics(args.data_path, 
                    num_samples_to_analyze=args.num_samples, 
                    num_radial_bins=args.radial_bins, 
                    num_angular_bins=args.angular_bins)
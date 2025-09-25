from io import BytesIO
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
from typing import Any
import tempfile
import shutil
import matplotlib.pyplot as plt

from utils.eval import create_dft_plot_artifact, create_correlation_plot_artifact, create_flipbook_artifact
from utils.fourier import FNetEncoderLayer, FNOTransformerLayer
from utils.custom_loss import K_losses

# helper function(s)
def get_padding_2d(input_shape, patch_size):
    """Calculate padding needed for height and width."""
    H, W = input_shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    # Pad evenly on both sides: (left, right, top, bottom)
    return (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

class F2FTransformer(pl.LightningModule):
    def __init__(
        self,
        # architecture specifics
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mcl_params: dict,
        # relevant hyperparameters 
        optimizer: Any,
        lr_scheduler: Any,
        near_field_dim: int,
        num_meta_atoms: int,
        loss_func: str,
        sample_idx: int
    ):  
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.mcl_params = mcl_params
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = lr_scheduler
        self.near_field_dim = near_field_dim
        self.num_meta_atoms = num_meta_atoms
        self.loss_func = loss_func
        self.sample_idx = sample_idx
        
        self.t_in_for_pos_embed = 1 # num of input steps pos embed should handle
        
        super().__init__()
        self.save_hyperparameters()
        
        # store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_step_outputs = []
        
        # setup architecture
        self.create_architecture()
        
    def create_architecture(self):
        I = 1 # number of parameters for 1 meta-atom (1 radius)
        N = self.num_meta_atoms # how many meta-atoms
        self.grid_size = (np.sqrt(self.num_meta_atoms), np.sqrt(self.num_meta_atoms))

        # --- Layers ---
        # 1. Patch Embedding: 1 design param (i.e. 1 of 9) -> embedding
        self.patch_embed = nn.Linear(I, self.embed_dim)

        # 2. Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, N, self.embed_dim)) # Spatial

        # 3. Transformer Block (Encoder Layers)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.embed_dim * self.mlp_ratio),
            dropout=self.dropout,
            activation=F.gelu,
            batch_first=True # Crucial for easier handling
        )
        # Use TransformerEncoder with attention
        self.transformer_block = nn.TransformerEncoder(transformer_layer, num_layers=self.depth)
        # LayerNorm before prediction head
        self.norm = nn.LayerNorm(self.embed_dim)
        # get attention weights for analysis
        self.attn_weights = None
        
        def get_attention_hook(module, input, output):
            self.attn_weights = output[1] # attn weights are the second element of the output tuple
            
        # register the hook on the self-attention module of the last layer
        self.transformer_block.layers[-1].self_attn.register_forward_hook(get_attention_hook)
    
        # convolution decoder for upsampling (3x3) -> (192x192)
        self.upsample_blocks = nn.Sequential(
            self._create_upsample_block(self.embed_dim, 512),
            self._create_upsample_block(512, 256),
            self._create_upsample_block(256, 128),
            self._create_upsample_block(128, 64),
            self._create_upsample_block(64, 32),
            self._create_upsample_block(32, 16)
        )
        self.final_conv = nn.Conv2d(16, 2, kernel_size=3, padding=1)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        
    # ------------------------
    # HELPER FUNCTIONS / LOGIC
    # ------------------------
      
    def _create_upsample_block(self, in_channels, out_channels):
        """Helper function to create a ConvTranspose2d block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU()
        )
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _embed_design(self, design):
        """Embeds a single design (B, N, D)."""
        # Linear(C, D) -> (B, N, D)
        patch_emb = self.patch_embed(design)

        # Add positional embedding (shape 1, N, D)
        patch_emb_spat = patch_emb + self.pos_embed # (B, N, D)

        # (B, N, D)
        return patch_emb_spat
    
    def _predict_field_from_embedding(self, embeddings):
        """Predicts a field (B, C, H, W) from the Transformer output embedding (B, N, D)."""
        B, N, D = embeddings.shape
        grid_H, grid_W = self.grid_size
        
        # 1. Reshape sequence back to 2D grid of embeddings
        # (B, N, D) -> (B, grid_H, grid_W, D)
        embeddings_grid = embeddings.view(B, grid_H, grid_W, D)
        
        # 2. Permute to match (B, C, H, W) for torch convs
        embeddings_permuted = embeddings_grid.permute(0, 3, 1, 2)
        
        # 3. Pass through conv decoder to upsample and generate field
        x = self.upsample_blocks(embeddings_permuted) # (B, 16, 192, 192)
        x_resized = F.interpolate(x, size=(166,166), mode='bilinear', align_corners=False)
        predicted_field = self.final_conv(x_resized) # (B, 2, 166, 166)
        
        return predicted_field
    
    # ----------------
    # STANDARD METHODS
    # ----------------
            
    def forward(self, designs):
        """forward pass mapping designs to near fields"""
        # 1. Embed design params into sequence of tokens
        design_embeddings = self._embed_design(designs) # (B, N, D)
        
        # 2. Process through encoder
        processed_embdeddings = self.transformer_block(src=design_embeddings)
        processed_embdeddings = self.norm(processed_embdeddings)
        
        # 3. Decode into the final field map
        predicted_fields = self._predict_field_from_embedding(processed_embdeddings)
        
        return predicted_fields
        
    def compute_loss(self, preds, labels, choice='mse'):
        """
        Compute loss given predictions and labels.
        """
        B, C, H, W = preds.shape
        
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
            
        elif choice == 'psnr':
            # Peak Signal-to-Noise Ratio
            preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
            fn = PeakSignalNoiseRatio(data_range=1.0).to(self._device)
            loss = fn(preds, labels)
            
        elif choice == 'gdl':
            preds_reshaped = preds.contiguous().view(B, C, H, W)
            labels_reshaped = labels.contiguous().view(B, C, H, W)

            # Gradient Difference Loss
            input_grad_v = preds_reshaped.diff(dim=-2) # Height dim
            target_grad_v = labels_reshaped.diff(dim=-2)
            input_grad_h = preds_reshaped.diff(dim=-1) # width dim
            target_grad_h = labels_reshaped.diff(dim=-1)
            
            # compute squared differences of the gradients
            loss_v = (input_grad_v - target_grad_v)**2
            loss_h = (input_grad_h - target_grad_h)**2
            
            # loss is the mean of all pixel-wise gradient differences
            loss = loss_v.mean() + loss_h.mean()

        elif choice == 'kspace': # the multi-param complex loss term chiefly controlled by mcl_params
            # convert complex tensors [B*T, H, W]
            pred_cplx = torch.complex(preds[:, 0, :, :], preds[:, 1, :, :])
            label_cplx = torch.complex(labels[:, 0, :, :], labels[:, 1, :, :])
            
            # commpute k-space loss terms
            loss_obj = K_losses(label_cplx, pred_cplx, num_bins=100)
            kMag = loss_obj.kMag(option='log')
            kPhase = loss_obj.kPhase(option='mag_weight')
            kRadial = loss_obj.kRadial()
            kAngular = loss_obj.kAngular()
            
            # compute the final compound loss
            # TODO: weightings
            loss = (kMag + kPhase + kRadial + kAngular)

        elif choice == 'ssim': # standard (full volume) 
            # Compute SSIM for each channel separately
            torch.use_deterministic_algorithms(True, warn_only=True)
            with torch.backends.cudnn.flags(enabled=False):
                ssim_vals = []
                for c in range(C):
                    pred_c = preds[:, c:c+1]  # Keep channel dimension
                    label_c = labels[:, c:c+1]
                    fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                    ssim_value = fn(pred_c, label_c)
                    ssim_vals.append(ssim_value)
                
            # Average SSIM across channels
            loss = 1 - torch.stack(ssim_vals).mean()
        
        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def objective(self, preds, labels):
        """
        objective loss calculation for the transformer.
        """
        # construct multi-criteria loss if requested
        if self.loss_func == "multi":
            # TODO: dynamically avoid the computation if its mcl param is 0?
            mse_loss = self.compute_loss(preds, labels, choice='mse') * self.mcl_params['alpha']
            ssim_loss = self.compute_loss(preds, labels, choice='ssim') * self.mcl_params['beta']
            gdl_loss = self.compute_loss(preds, labels, choice='gdl') * self.mcl_params['gamma']
            k_loss = self.compute_loss(preds, labels, choice='kspace') * self.mcl_params['delta']
            total_loss = mse_loss + ssim_loss + gdl_loss + k_loss
            
            return {"loss": total_loss, 
                    "mse": mse_loss, 
                    "ssim": ssim_loss,
                    "gdl": gdl_loss,
                    "kspace": k_loss}  
        
        else: # just do the one
            total_loss = self.compute_loss(preds, labels, choice=self.loss_func)
        
            return {"loss": total_loss}
    
    def configure_optimizers(self):
        """
        Setup optimzier and LR scheduler.
        """
        #optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
        #lr_scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer)
        optimizer = self.optimizer_cfg(params=self.parameters())
        lr_scheduler = self.scheduler_cfg(optimizer=optimizer)
        
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1,
                               "monitor": "val_loss"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def shared_step(self, batch, batch_idx):
        """
        logic for training/validation/testing steps using the Transformer.
        """
        designs, target_fields = batch # (B, N, C_in), (B, C_out, H, W)
        pred_fields = self.forward(designs) # (B, C_out, H, W)
        loss_dict = self.objective(pred_fields, target_fields)

        return loss_dict, pred_fields
        
    def training_step(self, batch, batch_idx):
        """
        training
        """
        loss_dict, preds = self.shared_step(batch, batch_idx)
        
        self.log("train_loss", loss_dict['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if self.loss_func == "multi": # log addl components
            self.log("train_mse", loss_dict['mse'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_kspace", loss_dict['kspace'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_gdl", loss_dict['gdl'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        # other metrics
        psnr_vals = []
        ssim_vals = []
        
        # Handle real and imaginary components separately
        for comp in range(2):
            pred_comp = preds[:, comp].unsqueeze(1)  # [B, 1, H, W]
            batch_comp = batch[:, comp].unsqueeze(1)  # [B, 1, H, W]
            
            psnr_vals.append(self.train_psnr(pred_comp.float(), batch_comp.float()))
            ssim_vals.append(self.train_ssim(pred_comp.float(), batch_comp.float()))
        
        psnr = torch.stack(psnr_vals).mean()
        ssim = torch.stack(ssim_vals).mean()
    
        self.log("train_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_ssim", ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss_dict['loss'], 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        """
        validation
        """
        loss_dict, preds = self.shared_step(batch, batch_idx)
        loss_dict, preds = self.shared_step(batch, batch_idx)
        
        self.log("val_loss", loss_dict['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if self.loss_func == "multi": # log additional components
            self.log("val_mse", loss_dict['mse'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_kspace", loss_dict['kspace'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_gdl", loss_dict['gdl'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        # Non-sequential case
        psnr_vals = []
        ssim_vals = []
        
        # Handle real and imaginary components separately
        for comp in range(2):
            pred_comp = preds[:, comp].unsqueeze(1)  # [B, 1, H, W]
            batch_comp = batch[:, comp].unsqueeze(1)  # [B, 1, H, W]
            
            psnr_vals.append(self.val_psnr(pred_comp.float(), batch_comp.float()))
            ssim_vals.append(self.val_ssim(pred_comp.float(), batch_comp.float()))
        
        psnr = torch.stack(psnr_vals).mean()
        ssim = torch.stack(ssim_vals).mean()
        
        self.log("val_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_ssim", ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss_dict['loss'], 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Good ol testing step
        """
        loss_dict, preds = self.shared_step(batch, batch_idx)
        loss_dict, preds = self.shared_step(batch, batch_idx)
        samples, labels = batch
        
        # log the loss for this batch
        self.log("test_loss_step", loss_dict['loss'], on_step=True, on_epoch=False, prog_bar=False)
        
        # keep preds and labels for aggregation
        output = {"preds": preds.detach().cpu().numpy(), "labels": labels.detach().cpu().numpy()}
        self.test_step_outputs.append(output)
        return output
    
    '''def on_validation_epoch_end(self):
        if self.attn_weights is None:
            return
        
        # grab a single sample from the batch -> (num_heads, seq_len, seq_len)
        attn_map = self.attn_weights[0].detach().cpu()
        
        # average the attention weights across all heads
        attn_map = attn_map.mean(dim=0)
        
        cls_attn_map = attn_map[0, 1:]
        
        # reshape 1D patch attention vector into 2D grid
        grid_h, grid_w = self.grid_size
        # ensure num patches matches
        if cls_attn_map.shape[0] == grid_h * grid_w:
            img_map = cls_attn_map.reshape(grid_h, grid_w)
        else:
            print("Attention map size does not match grid size")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_map, cmap='viridis', interpolation='nearest')
        ax.set_title(f"Attention Map from [CLS] Token at Epoch {self.current_epoch}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # logging to mlflow
        self.log_matplotlib_figure(fig, "attention_map_evolution")

        # clear the stored weights for the next validation pass
        self.attn_weights = None
        
    def log_matplotlib_figure(self, fig, artifact_name):
        """Logs a matplotlib figure to MLflow."""
        # Save the plot to a buffer
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        
        # Get the MLflow logger client
        mlflow_logger = self.logger.experiment
        
        try:
            # Log the image from the buffer
            mlflow_logger.log_image(
                self.logger.run_id,
                buf,
                f"{artifact_name}.png",
                # step to create slider
                step=self.current_epoch
            )
        except Exception as e:
            print(f"Failed to log image to MLflow: {e}")

        # Close the figure to free memory
        plt.close(fig)'''
    
    def on_test_epoch_end(self):
        # calculate and log aggregate metrics
        all_preds = np.concatenate([x['preds'] for x in self.test_step_outputs], axis=0)
        all_labels = np.concatenate([x['labels'] for x in self.test_step_outputs], axis=0)
        
        # free memory
        self.test_step_outputs.clear()
        
        temp_dir = tempfile.mkdtemp()
        try:
            mlflow_client = self.logger.experiment
            run_id = self.logger.run_id
            
            plot_results_dict = {"target": all_labels, "predictions": all_preds}
            create_dft_plot_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir, sample_idx=self.sample_idx)
            create_correlation_plot_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir)
            create_flipbook_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir, sample_idx=self.sample_idx)
            
            mlflow_client.log_artifacts(run_id, temp_dir, artifact_path="evaluation_plots")
            print("--- Artifacts logged successfully to the corresponding run folder. ---")
            print("--- Artifacts logged successfully to the corresponding run folder. ---")

        finally:
            shutil.rmtree(temp_dir)
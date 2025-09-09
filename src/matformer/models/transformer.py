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

class NewWaveTransformer(pl.LightningModule):
    def __init__(
        self,
        # architecture specifics
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mixing: str,
        num_blocks: int,
        dropout: float,
        mcl_params: dict,
        # relevant hyperparameters 
        optimizer: Any,
        lr_scheduler: Any,
        near_field_dim: int,
        loss_func: str,
        seq_len: int
    ):  
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mixing = mixing
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.mcl_params = mcl_params
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = lr_scheduler
        self.near_field_dim = near_field_dim
        self.loss_func = loss_func
        self.seq_len = seq_len
        
        self.t_in_for_pos_embed = 1 # num of input steps pos embed should handle
        self.max_steps = self.t_in_for_pos_embed + self.seq_len
        
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
        
    def create_architecture(self):
        C = 2 # Input channels (real, imag)
        H, W = self.near_field_dim, self.near_field_dim
        padding_dims = get_padding_2d((H, W), self.patch_size)
        self.padded_H = H + padding_dims[2] + padding_dims[3]
        self.padded_W = W + padding_dims[0] + padding_dims[1]
        self.grid_size = (self.padded_H // self.patch_size, self.padded_W // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        N = self.num_patches

        # --- Layers ---
        # 1. Patch Embedding (used for input and feedback)
        self.patch_embed = nn.Conv2d(C, self.embed_dim,
                                     kernel_size=self.patch_size,
                                     stride=self.patch_size)

        # 2. Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, N, self.embed_dim)) # Spatial
        self.temporal_embed = nn.Parameter(torch.zeros(1, self.max_steps, self.embed_dim)) # Temporal

        # 3. Transformer Block (Encoder Layers)
        if self.mixing == 'default':
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
            
        elif self.mixing == 'fnet':
            fnet_layers = [FNetEncoderLayer(self.embed_dim, self.mlp_ratio, self.dropout) for _ in range(self.depth)]
            self.transformer_block = nn.Sequential(*fnet_layers)
            #self.norm = nn.LayerNorm(self.embed_dim)
            
        elif self.mixing == 'afno':
            fno_layers = [FNOTransformerLayer(self.embed_dim, self.num_blocks, self.mlp_ratio, self.dropout) for _ in range(self.depth)]
            self.transformer_block = nn.Sequential(*fno_layers)

        # 4. Prediction Head
        # Takes the final embedding (D) and predicts all patch features (N*P*P*C)
        # Needs to reconstruct the full spatial frame from a single vector per time step
        '''bottleneck_channels = self.embed_dim # e.g., 256
        up_channels = [bottleneck_channels, bottleneck_channels // 2, bottleneck_channels // 4, bottleneck_channels // 8] # e.g., [256, 128, 64, 32]
        if up_channels[-1] < C: # Ensure last channel count is at least C
             up_channels[-1] = C

        # Project D embedding to start the spatial grid (grid_H x grid_W)
        self.head_bottleneck_proj = nn.Linear(self.embed_dim, up_channels[0] * self.grid_size[0] * self.grid_size[1])

        # Upsampling blocks (4 stages to go from 11x11 -> 176x176)
        self.upsample_block1 = self._create_upsample_block(up_channels[0], up_channels[1]) # 11x11 -> 22x22
        self.upsample_block2 = self._create_upsample_block(up_channels[1], up_channels[2]) # 22x22 -> 44x44
        self.upsample_block3 = self._create_upsample_block(up_channels[2], up_channels[3]) # 44x44 -> 88x88
        # Final block goes to C=2 channels aannd no batchnorm needed prob
        self.upsample_block4 = nn.ConvTranspose2d(up_channels[3], C, kernel_size=4, stride=2, padding=1) # 88x88 -> 176x176'''
        
        # MLP PREDICTION HEAD
        self.prediction_head_mlp = nn.Sequential(
            # Map D -> N*D (Generate embeddings for all patch locations)
            nn.Linear(self.embed_dim, N * self.embed_dim),
            nn.GELU()
        )
        # Project each generated patch embedding D -> P*P*C
        self.patch_projection = nn.Linear(self.embed_dim, self.patch_size * self.patch_size * C)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.temporal_embed, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _embed_frame(self, frame, t_step):
        """Embeds a single frame (B, C, H, W) at time step t_step."""
        B, C, H, W = frame.shape
        N = self.num_patches
        D = self.embed_dim

        padding = get_padding_2d((H, W), self.patch_size)
        frame_padded = F.pad(frame, padding) # (B, C, H_pad, W_pad)

        # Patch embed -> Flatten -> Permute
        patch_emb = self.patch_embed(frame_padded).flatten(2).permute(0, 2, 1) # (B, N, D)

        # Add spatial embedding
        patch_emb_spat = patch_emb + self.pos_embed # (B, N, D)

        # Add temporal embedding for the given step
        patch_emb_spat_temp = patch_emb_spat + self.temporal_embed[:, t_step, :].unsqueeze(1) # (B, N, D)

        # Reshape for sequence concatenation: (B, 1*N, D)
        return patch_emb_spat_temp.view(B, N, D).reshape(B, 1 * N, D)
    
    def _generate_square_subsequent_mask(self, sz, device):
        """Generates causal mask for sz x sz."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _predict_frame_from_embedding(self, embedding):
        """Predicts a frame (B, C, H, W) from the Transformer output embedding (B, D)."""
        B, D = embedding.shape
        N = self.num_patches
        C = 2 # Assuming C=2
        H, W = self.near_field_dim, self.near_field_dim
        P = self.patch_size
        grid_H, grid_W = self.grid_size

        # 1. Project D -> N*D
        projected_patches = self.prediction_head_mlp(embedding) # (B, N*D)

        # 2. Reshape to (B, N, D)
        patch_embeddings = projected_patches.view(B, N, D)

        # 3. Project each patch D -> P*P*C
        # Input (B, N, D) -> Reshape (B*N, D) -> Linear -> (B*N, P*P*C)
        patch_pixels_flat = self.patch_projection(patch_embeddings.view(-1, D)) # (B*N, P*P*C)

        # 4. Unpatch: Reshape and permute to form image grid
        # -> (B, N, P*P*C)
        patch_pixels_flat = patch_pixels_flat.view(B, N, -1)
        # -> (B, grid_H, grid_W, P*P*C)
        patch_pixels_grid = patch_pixels_flat.view(B, grid_H, grid_W, P*P*C)
        # -> (B, grid_H, grid_W, P, P, C)
        patch_pixels_6d = patch_pixels_grid.view(B, grid_H, grid_W, P, P, C)
        # -> Permute (B, C, grid_H, P, grid_W, P)
        patch_pixels_perm = patch_pixels_6d.permute(0, 5, 1, 3, 2, 4)
        # -> Reshape (B, C, H_pad, W_pad)
        frame_padded = patch_pixels_perm.reshape(B, C, self.padded_H, self.padded_W)

        # 5. Crop padding
        padding = get_padding_2d((H, W), P)
        crop_h_start = padding[2]; crop_h_end = self.padded_H - padding[3]
        crop_w_start = padding[0]; crop_w_end = self.padded_W - padding[1]
        frame = frame_padded[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        return frame # (B, C, H, W)
    
    # ----------------
    # STANDARD METHODS
    # ----------------
            
    def forward(self, x, labels=None, meta=None):
        """
        Autoregressive forward pass. Uses teacher forcing during training.

        Args:
            x (torch.Tensor): Input frames (B, T_in, C, H, W). Assumes T_in=1 for now.
            labels (torch.Tensor, optional): Ground truth labels (B, T_out, C, H, W).
                                            Used for teacher forcing during training.
        Returns:
            torch.Tensor: Predicted sequence (B, T_out, C, H, W)
        """
        #print(f"x.shape: {x.shape}")
        B, T_in, C, H, W = x.shape
        T_out = self.seq_len
        N = self.num_patches
        D = self.embed_dim
        device = x.device

        # Assert T_in == 1 for simplicity in this implementation
        if T_in != 1 and self.t_in_for_pos_embed != T_in:
             print(f"[Warning] T_in={T_in} but expecting {self.t_in_for_pos_embed}. Adjust config/model if needed.")
             # For now, proceed assuming T_in=1 for embedding indexing logic
             T_in = 1 # Override T_in for indexing consistency if mismatch

        if self.training and labels is not None:
            # --- Teacher Forcing during Training ---
            # 1. Prepare Input Sequence
            target_input_frames = torch.cat([x, labels[:, :-1, ...]], dim=1) # (B, T_out, C, H, W)
            T = target_input_frames.shape[1] # T = T_out (assuming T_in=1)

            # 2. Embed Sequence -> full_seq_embeddings (B, T*N, D)
            embedded_tokens_list = []
            for t in range(T):
                # _embed_frame uses parameters (patch_embed, pos_embed, temporal_embed)
                embedded_tokens_list.append(self._embed_frame(target_input_frames[:, t, ...], t))
            full_seq_embeddings = torch.cat(embedded_tokens_list, dim=1) # Concatenation should be fine
            # At this point, full_seq_embeddings *should* require grad due to embedding parameters.

            # 3. Create Causal Mask
            attn_mask = self._generate_square_subsequent_mask(T * N, device) # Does not require grad

            # 4. Pass through Transformer Block and norm
            if self.mixing == 'default':
                transformer_out = self.transformer_block(
                    src=full_seq_embeddings,
                    mask=attn_mask
                )

                # normalize
                transformer_out_norm = self.norm(transformer_out)
                
            elif self.mixing in ['fnet', 'afno']:
                # norm is already handled implicitly
                transformer_out_norm = self.transformer_block(full_seq_embeddings)

            # 5. Extract Output Embeddings for Prediction
            # T should be T_out here
            output_embeddings = transformer_out_norm.view(B, T, N, D)[:, :, -1, :] # Slicing/view *should* preserve grad. Output *should* require grad.

            # 6. Generate Predictions
            # _predict_frame_from_embedding uses head parameters.
            # Input `output_embeddings.reshape(-1, D)` *should* require grad.
            preds = self._predict_frame_from_embedding(output_embeddings.reshape(-1, D))
            preds = preds.view(B, T_out, C, H, W) # Reshape back. `preds` *should* require grad.

            return preds

        else:
            # --- Autoregressive Generation during Inference ---
            generated_frames = []
            # Embed the initial input frame (step 0)
            current_seq_embeddings = self._embed_frame(x[:, 0, ...], 0) # (B, 1*N, D)

            # Ensure no gradients are computed during generation loop
            with torch.no_grad():
                for t in range(T_out):
                    # Current sequence length feeding into transformer
                    current_seq_len_tokens = current_seq_embeddings.shape[1] # (T_in + t) * N

                    # Create causal mask for the current sequence length
                    attn_mask = self._generate_square_subsequent_mask(current_seq_len_tokens, device)

                    # Pass current sequence through Transformer
                    if self.mixing == 'default':
                        transformer_out = self.transformer_block(src=current_seq_embeddings, mask=attn_mask) # (B, current_seq_len_tokens, D)
                        transformer_out_norm = self.norm(transformer_out) # Normalize
                    elif self.mixing in ['fnet', 'afno']:
                        transformer_out_norm = self.transformer_block(current_seq_embeddings)

                    # Get embedding corresponding to the *last* input token
                    last_embedding = transformer_out_norm[:, -1, :] # (B, D)

                    # Predict the next frame (frame t+1 in 0-based indexing, or frame t in sequence gen)
                    pred_frame_t = self._predict_frame_from_embedding(last_embedding) # (B, C, H, W)
                    generated_frames.append(pred_frame_t)

                    # If not the last frame to predict, embed it and append
                    if t < T_out - 1:
                        # Embed the predicted frame for the *next* time step index (t+1)
                        # The temporal embedding index should be t+1 (assuming T_in=1)
                        next_token_embeddings = self._embed_frame(pred_frame_t, t + T_in) # (B, 1*N, D)

                        # Append to the sequence for the next iteration
                        current_seq_embeddings = torch.cat([current_seq_embeddings, next_token_embeddings], dim=1)

            # Stack generated frames
            preds = torch.stack(generated_frames, dim=1) # (B, T_out, C, H, W)
            return preds
        
    def compute_loss(self, preds, labels, choice='mse'):
        """
        Compute loss given predictions and labels.
        """
        if preds.ndim == 3: # vanilla LSTM, for example, flattens spatial and r/i dims
            preds = preds.view(preds.shape[0], preds.shape[1], 2, self.near_field_dim, self.near_field_dim)
            labels = labels.view(labels.shape[0], labels.shape[1], 2, self.near_field_dim, self.near_field_dim)
        B, T, C, H, W = preds.shape

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
            preds_reshaped = preds.contiguous().view(B * T, C, H, W)
            labels_reshaped = labels.contiguous().view(B * T, C, H, W)

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
            preds_reshaped = preds.view(B*T, C, H, W)
            labels_reshaped = labels.view(B*T, C, H, W)

            # convert complex tensors [B*T, H, W]
            pred_cplx = torch.complex(preds_reshaped[:, 0, :, :], preds_reshaped[:, 1, :, :])
            label_cplx = torch.complex(labels_reshaped[:, 0, :, :], labels_reshaped[:, 1, :, :])
            
            # commpute k-space loss terms
            loss_obj = K_losses(label_cplx, pred_cplx, num_bins=100)
            kMag = loss_obj.kMag(option='log')
            kPhase = loss_obj.kPhase(option='mag_weight')
            kRadial = loss_obj.kRadial()
            kAngular = loss_obj.kAngular()
            
            # compute the final compound loss
            loss = (self.mcl_params['alpha'] * kMag + self.mcl_params['beta'] * kPhase +
                            self.mcl_params['gamma'] * kRadial + self.mcl_params['delta'] * kAngular)

        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def objective(self, preds, labels):
        """
        objective loss calculation for the transformer.
        """
        # construct multi-criteria loss if requested
        if self.loss_func == "multi":
            mse_loss = self.compute_loss(preds, labels, choice='mse')
            gdl_loss = self.compute_loss(preds, labels, choice='gdl')
            total_loss = mse_loss + self.mcl_params['alpha'] * gdl_loss
            
            return {"loss": total_loss, "mse": mse_loss, "gdl": gdl_loss}
        
        if self.loss_func == "kspace":
            mse_loss = self.compute_loss(preds, labels, choice='mse')
            k_loss = self.compute_loss(preds, labels, choice='kspace')
            total_loss = mse_loss + k_loss
            
            return {"loss": total_loss, "mse": mse_loss, "kspace": k_loss}

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
        samples, labels = batch # samples shape (B, T_in, C, H, W), labels shape (B, T_out, C, H, W)
        preds = self.forward(samples, labels) # preds shape (B, T_out, C, H, W)
        loss_dict = self.objective(preds, labels)

        return loss_dict, preds
        
    def training_step(self, batch, batch_idx):
        """
        training
        """
        loss_dict, preds = self.shared_step(batch, batch_idx)
        
        self.log("train_loss", loss_dict['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if self.loss_func == "multi":
            self.log("train_mse", loss_dict['mse'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_gdl", loss_dict['gdl'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if self.loss_func == "kspace":
            self.log("train_mse", loss_dict['mse'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_kspace", loss_dict['kspace'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        # For sequential models, we need to handle multiple timesteps
        if isinstance(batch, list):
            # Get the ground truth sequence
            truth = batch[1]  # target sequence
            
            # Compute PSNR/SSIM for each timestep and average
            psnr_vals = []
            ssim_vals = []
            
            # Ensure predictions and truth have same number of timesteps
            for t in range(min(preds.shape[1], truth.shape[1])):
                pred_t = preds[:, t]  # [B, 2, H, W]
                truth_t = truth[:, t]  # [B, 2, H, W]
                
                # Handle real and imaginary components separately
                for comp in range(2):
                    pred_comp = pred_t[:, comp].unsqueeze(1)  # [B, 1, H, W]
                    truth_comp = truth_t[:, comp].unsqueeze(1)  # [B, 1, H, W]
                    
                    # Compute metrics for this component and timestep
                    psnr_t = self.train_psnr(pred_comp.float(), truth_comp.float())
                    ssim_t = self.train_ssim(pred_comp.float(), truth_comp.float())
                    
                    psnr_vals.append(psnr_t)
                    ssim_vals.append(ssim_t)
            
            # Average metrics across timesteps and components
            psnr = torch.stack(psnr_vals).mean()
            ssim = torch.stack(ssim_vals).mean()
            
        else:
            # Non-sequential case
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
        
        self.log("val_loss", loss_dict['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if self.loss_func == "multi":
            self.log("val_mse", loss_dict['mse'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_gdl", loss_dict['gdl'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if self.loss_func == "kspace":
            self.log("val_mse", loss_dict['mse'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_kspace", loss_dict['kspace'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        # For sequential models, we need to handle multiple timesteps
        if isinstance(batch, list):
            # Get the ground truth sequence
            truth = batch[1] # target seq
            
            # Compute PSNR/SSIM for each timestep and average
            psnr_vals = []
            ssim_vals = []
            
            # Ensure predictions and truth have same number of timesteps
            for t in range(min(preds.shape[1], truth.shape[1])):
                pred_t = preds[:, t]  # [B, 2, H, W]
                truth_t = truth[:, t]  # [B, 2, H, W]
                
                # Handle real and imaginary components separately
                for comp in range(2):
                    pred_comp = pred_t[:, comp].unsqueeze(1)  # [B, 1, H, W]
                    truth_comp = truth_t[:, comp].unsqueeze(1)  # [B, 1, H, W]
                    
                    # Compute metrics for this component and timestep
                    psnr_t = self.val_psnr(pred_comp.float(), truth_comp.float())
                    ssim_t = self.val_ssim(pred_comp.float(), truth_comp.float())
                    
                    psnr_vals.append(psnr_t)
                    ssim_vals.append(ssim_t)
            
            # Average metrics across timesteps and components
            psnr = torch.stack(psnr_vals).mean()
            ssim = torch.stack(ssim_vals).mean()
            
        else:
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
        samples, labels = batch
        
        # log the loss for this batch
        self.log("test_loss_step", loss_dict['loss'], on_step=True, on_epoch=False, prog_bar=False)
        
        # keep preds and labels for aggregation
        output = {"preds": preds.detach().cpu().numpy(), "labels": labels.detach().cpu().numpy()}
        self.test_step_outputs.append(output)
        return output
    
    def on_validation_epoch_end(self):
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
        plt.close(fig)
    
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
            create_dft_plot_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir, sample_idx=0)
            create_correlation_plot_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir)
            create_flipbook_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir, sample_idx=0)
            
            mlflow_client.log_artifacts(run_id, temp_dir, artifact_path="evaluation_plots")
            print("--- Artifacts logged successfully to the corresponding run folder. ---")

        finally:
            shutil.rmtree(temp_dir)
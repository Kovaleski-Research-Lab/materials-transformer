import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
import hydra
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any

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
        use_diff_loss: bool,
        lambda_diff: float,
        dropout: float,
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
        self.use_diff_loss = use_diff_loss
        self.lambda_diff = lambda_diff
        self.dropout = dropout
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
        # LayerNorm before prediction head is common
        self.norm = nn.LayerNorm(self.embed_dim)

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
            Tuple[torch.Tensor, None]: Predicted sequence (B, T_out, C, H, W), None
        """
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

            # 4. Pass through Transformer Block
            transformer_out = self.transformer_block(
                src=full_seq_embeddings,
                mask=attn_mask
            ) # Uses transformer parameters. Output *should* require grad.

            # 5. Normalize
            transformer_out_norm = self.norm(transformer_out) # Uses norm parameters. Output *should* require grad.

            # 6. Extract Output Embeddings for Prediction
            # T should be T_out here
            output_embeddings = transformer_out_norm.view(B, T, N, D)[:, :, -1, :] # Slicing/view *should* preserve grad. Output *should* require grad.

            # 7. Generate Predictions
            # _predict_frame_from_embedding uses head parameters.
            # Input `output_embeddings.reshape(-1, D)` *should* require grad.
            preds = self._predict_frame_from_embedding(output_embeddings.reshape(-1, D))
            preds = preds.view(B, T_out, C, H, W) # Reshape back. `preds` *should* require grad.

            return preds, None

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
                    transformer_out = self.transformer_block(src=current_seq_embeddings, mask=attn_mask) # (B, current_seq_len_tokens, D)
                    transformer_out_norm = self.norm(transformer_out) # Normalize

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
            return preds, None
        
    def compute_loss(self, preds, labels, choice='mse'):
        """
        Compute loss given predictions and labels.
        Subclasses can override if needed, but this base implementation is standard.
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

        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def objective(self, preds, labels):
        """
        objective loss calculation for the transformer. Differs from other implementations in the
        inclusion of an optional difference loss term for temporal dynamics.
        """
        use_diff_loss = self.use_diff_loss
        lambda_diff = self.lambda_diff

        # --- 1. Calculate the base loss ---
        base_loss = self.compute_loss(preds, labels, choice=self.loss_func)
        total_loss = base_loss

        # --- 2. Calculate the difference loss (if enabled) ---
        if use_diff_loss and preds.shape[1] > 1: # sequence length needs to be greater than 1
            print("[DEBUG] Using diff loss")
            diff_loss = torch.tensor(0.0, device=preds.device, requires_grad=True) # Initialize as zero tensor on correct device
            # Calculate differences between consecutive time steps
            # preds shape: (B, T, C, H, W)
            pred_diff = preds[:, 1:, ...] - preds[:, :-1, ...]  # Shape: (B, T-1, C, H, W)
            label_diff = labels[:, 1:, ...] - labels[:, :-1, ...] # Shape: (B, T-1, C, H, W)

            # Calculate MSE loss on the differences.
            diff_loss_fn = nn.MSELoss()
            diff_loss = diff_loss_fn(pred_diff, label_diff)

            # Combine the base loss and the weighted difference loss
            total_loss = base_loss + lambda_diff * diff_loss

        '''# --- 3. Logging ---
        log_prefix = 'train' if self.training else 'val'

        self.log(f'{log_prefix}/base_loss', base_loss,
                 on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        if use_diff_loss and preds.shape[1] > 1:
            self.log(f'{log_prefix}/diff_loss', diff_loss,
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            # Log the lambda value used, helpful for tracking experiments
            self.log('hyperparameters/lambda_diff', lambda_diff,
                     on_step=False, on_epoch=True, sync_dist=True)'''
 
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
        Shared logic for training/validation steps using the Transformer.
        """
        samples, labels = batch # samples shape (B, T_in, C, H, W), labels shape (B, T_out, C, H, W)
        preds, _ = self.forward(samples, labels) # preds shape (B, T_out, C, H, W)
        loss_dict = self.objective(preds, labels)
        loss = loss_dict['loss']

        return loss, preds

    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        samples, labels = batch
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Determine the mode based on dataloader_idx
        if dataloader_idx == 0:
            mode = 'valid'
        elif dataloader_idx == 1:
            mode = 'train'
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        
        # Append predictions
        self.test_results[mode]['nf_pred'].append(preds_np)
        self.test_results[mode]['nf_truth'].append(labels_np)
        
    def training_step(self, batch, batch_idx):
        """
        Common training step shared among all subclasses
        """
        loss, preds = self.shared_step(batch, batch_idx)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
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
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        """
        Common validation step shared among all subclasses
        """
        loss, preds = self.shared_step(batch, batch_idx)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
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
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Common testing step likely shared among all subclasses
        """
        loss, preds = self.shared_step(batch, batch_idx)
        self.organize_testing(preds, batch, batch_idx, dataloader_idx)
    
    def on_test_end(self):
        """
        After testing, this method compiles results and logs them.
        """
        for mode in ['train', 'valid']:
            if self.test_results[mode]['nf_pred']:
                self.test_results[mode]['nf_pred'] = np.concatenate(self.test_results[mode]['nf_pred'], axis=0)
                self.test_results[mode]['nf_truth'] = np.concatenate(self.test_results[mode]['nf_truth'], axis=0)
                
            else:
                print(f"No test results for mode: {mode}")
        
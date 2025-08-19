from io import BytesIO
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
from typing import Any

#from utils.eval import create_dft_plot_artifact, create_correlation_plot_artifact, create_flipbook_artifact
#from utils.fourier import FNetEncoderLayer, FNOTransformerLayer

class ModeMLP(pl.LightningModule):
    def __init__(
        self,
        layers: list,
        activation: str,
        dropout: float,
        # base hyperparams
        optimizer: Any,
        lr_scheduler: Any,
        num_modes: int,
        num_design_conf: int,
        loss_func: str
    ):
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = lr_scheduler
        self.num_modes = num_modes
        self.num_design_conf = num_design_conf
        self.loss_func = loss_func
        
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
        layers = []
        in_features = self.num_modes
        for layer_size in self.layers:
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(nn.Dropout(self.dropout))
            layers.append(self.get_activation_function(self.activation))
            in_features = layer_size
            layers.append(nn.Linear(in_features, self.output_size))
        self.mlp = nn.Sequential(*layers)
    
    def get_activation_function(self, activation_name):
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        
    def compute_loss(self, preds, labels, choice):
        """
        Compute loss given predictions and labels.
        """
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        elif choice == 'emd':
            # ignoring emd for now
            raise NotImplementedError("Earth Mover's Distance not implemented!")
        elif choice == 'psnr':
            # Peak Signal-to-Noise Ratio
            preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
            fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            psnr_value = fn(preds, labels)
            loss = -psnr_value # minimize negative psnr
        elif choice == 'ssim':
            # Structural Similarity Index
            if preds.size(-1) < 11 or preds.size(-2) < 11:
                loss = 0 # if the size is too small, SSIM is not defined
            else:
                preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
                torch.use_deterministic_algorithms(True, warn_only=True)
                with torch.backends.cudnn.flags(enabled=False):
                    fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                    ssim_value = fn(preds, labels)
                    #print(f'SSIM VALUE: {ssim_value}')
                    ssim_comp = (1 - ssim_value)
                #loss = ssim_comp
                # Mean Squared Error
                preds = preds.to(torch.float32).contiguous()
                labels = labels.to(torch.float32).contiguous()
                fn2 = torch.nn.MSELoss()
                mse_comp = fn2(preds, labels)
                loss = self.conf.mcl_params['alpha'] * mse_comp + self.conf.mcl_params['beta'] * ssim_comp
                #loss = ssim_comp
        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def objective(self, preds, labels):
        loss = self.compute_loss(preds, labels, choice=self.loss_func)
        return {'loss': loss}
    
    def configure_optimizers(self):
        """
        Setup optimzier and LR scheduler.
        """
        optimizer = self.optimizer_cfg(params=self.parameters())
        lr_scheduler = self.scheduler_cfg(optimizer=optimizer)
        
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1,
                               "monitor": "val_loss"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        preds = self.forward(samples, labels)
        loss_dict = self.objective(preds, labels)
        loss = loss_dict['loss']
        return loss, preds
    
    def training_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch, batch_idx)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch, batch_idx)
        samples, labels = batch
        
        # log the loss for this batch
        self.log("test_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        
        # keep preds and labels for aggregation
        output = {"preds": preds.detach().cpu().numpy(), "labels": labels.detach().cpu().numpy()}
        self.test_step_outputs.append(output)
        return output
    
    def on_test_epoch_end(self):
        # calculate and log aggregate metrics
        all_preds = np.concatenate([x['preds'] for x in self.test_step_outputs], axis=0)
        all_labels = np.concatenate([x['labels'] for x in self.test_step_outputs], axis=0)
        
        # free memory
        self.test_step_outputs.clear()
        
        '''temp_dir = tempfile.mkdtemp()
        try:
            mlflow_client = self.logger.experiment
            run_id = self.logger.run_id
            
            plot_results_dict = {"target": all_labels, "predictions": all_preds}
            create_dft_plot_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir, sample_idx=0)
            create_correlation_plot_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir)
            create_flipbook_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir, sample_idx=0)
            
            mlflow_client.log_artifacts(run_id, temp_dir, artifact_path="evaluation_plots")
            print(f"--- Artifacts logged successfully to the corresponding run folder. ---")

        finally:
            shutil.rmtree(temp_dir)'''
from io import BytesIO
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch_geometric.nn import TransformerConv
import numpy as np
from typing import Any
import tempfile
import shutil
import matplotlib.pyplot as plt
import math

from utils.eval import create_matrix_artifact

ATOM_MAP = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'S': 4, 'Cl': 5, 'PAD': 6}

# Helper class for Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        # model_dim is the feature dimension of the input to the transformer
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Adjust shape for batch_first or not
        if batch_first:
            pe = pe.unsqueeze(0) # Shape: (1, max_len, model_dim)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, model_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, model_dim) if batch_first
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return x

class IRTransformer(pl.LightningModule):
    def __init__(
        self,
        # arch specs
        input_dim: int,
        model_dim: int,
        num_heads: int,
        depth: int,
        rows: int,
        cols: int,
        dropout: float,
        optimizer: Any,
        lr_scheduler: Any,
        loss_func: str,
        seq_len: int,
        encoder_depth: int,
        decoder_depth: int,
        target_vocab_size: int
    ):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.rows = rows
        self.cols = cols
        self.dropout = dropout
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = lr_scheduler
        self.loss_func = loss_func
        self.seq_len = seq_len # IR spectrum length
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.target_vocab_size = target_vocab_size
        
        super().__init__()
        self.save_hyperparameters()
        
        # ===== 1. ENCODER =====
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, batch_first=True)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.encoder_depth)
        
        # ===== 2. DECODER =====
        # embedding for the SMILES character tokens
        self.smiles_embedding = nn.Embedding(target_vocab_size, model_dim)
        # the decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.decoder_depth)
        
        
        # ===== 3. FINAL OUTPUT LAYER =====
        self.output_fc = nn.Linear(model_dim, target_vocab_size)
        
        # store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_step_outputs = []
        
    def forward(self, src_spectrum, trg_smiles):
        # 1. Encode the IR spectrum
        src_embedded = self.encoder_embedding(src_spectrum)
        src_encoded = self.pos_encoder(src_embedded)
        memory = self.transformer_encoder(src_encoded) # context
        
        # 2. Prepare target sequence for the decoder
        trg_embedded = self.smiles_embedding(trg_smiles)
        trg_encoded = self.pos_encoder(trg_embedded)
        
        # 3. Decode
        decoder_output = self.transformer_decoder(tgt=trg_encoded, memory=memory)
        
        return self.output_fc(decoder_output)
    
    '''def compute_loss(self, preds, labels, choice="bce"):
        if choice == 'bce':
            # Binary Cross Entropy
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.BCEWithLogitsLoss()
            loss = fn(preds, labels)
            
        elif choice == 'mse':
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
        return {"loss": self.compute_loss(preds, labels, choice=self.loss_func)}'''
        
    def objective(self, batch):
        samples, labels = batch # labels should now be a tuple: (true_atom_types, true_adj_matrix)
        true_atom_types, true_adj_matrix = labels
        
        pred_atom_types, pred_adj_matrix = self.forward(samples)

        # --- Combined Loss ---
        # 1. Loss for predicting the correct atom types (Nodes)
        # Note: permute is needed as CrossEntropyLoss expects (N, C, ...)
        loss_nodes = F.cross_entropy(pred_atom_types.permute(0, 2, 1), true_atom_types)
        
        # 2. Loss for predicting the correct bonds (Edges)
        loss_edges = F.binary_cross_entropy_with_logits(pred_adj_matrix, true_adj_matrix)

        # Combine the losses (you can weight them if needed)
        total_loss = loss_nodes + loss_edges
        return {"loss": total_loss, "loss_nodes": loss_nodes, "loss_edges": loss_edges}
    
    def configure_optimizers(self):
        optimizer = self.optimizer_cfg(params=self.parameters())
        lr_scheduler = self.scheduler_cfg(optimizer=optimizer)
        
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1,
                               "monitor": "val_loss"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def shared_step(self, batch, batch_idx):
        spectrum, smiles_true = batch
        
        # input to decoder is the sequence minus <EOS>
        smiles_input = smiles_true[:, :-1]
        
        # target for loss is the seq minus <SOS>
        smiles_target = smiles_true[:, 1:]
        
        # get output
        logits = self.forward(spectrum, smiles_input)

        # calc loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            smiles_target.reshape(-1),
            ignore_index = self.pad_token_idx
        )
        self.log("train_loss", loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # Log the metrics
        self.log("train_loss", loss,
                    prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # Log the metrics
        self.log_dict("val_loss", loss, 
                    on_step=False, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def generate_smiles(self, spectrum, max_len=100, sos_idx=0, eos_idx=1):
        self.eval()
        # Encode the spectrum once
        memory = self.transformer_encoder(self.pos_encoder(self.encoder_embedding(spectrum)))
        
        # Start with the <SOS> token
        output_tokens = torch.tensor([sos_idx], dtype=torch.long, device=self.device)

        for _ in range(max_len - 1):
            # Get model prediction for the next token
            logits = self.forward(spectrum, output_tokens.unsqueeze(0))
            next_token = logits.argmax(dim=-1)[:, -1] # Get the last token
            
            # Append the prediction to our output
            output_tokens = torch.cat([output_tokens, next_token], dim=0)

            # Stop if we predict the <EOS> token
            if next_token.item() == eos_idx:
                break
                
        return output_tokens
    
    def test_step(self, batch, batch_idx):
        spectrum, smiles_true = batch

        # generate an output string
        smiles_pred = self.generate_smiles(spectrum)
        
        output = {
            "pred_tokens": smiles_pred.detach().cpu(),
            "true_tokens": smiles_true.detach().cpu()
        }
        self.test_step_outputs.append(output)
        
        return output
        
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("Test step outputs were empty. Skipping artifact generation.")
            return
        
        # from datamodule, get vocab
        tokenizer = self.trainer.datamodule.tokenizer
        
        true_smiles_list = []
        pred_smiles_list = []

        # Decode all token sequences back into strings
        for output in self.test_step_outputs:
            # Decode the true SMILES string
            true_smiles = tokenizer.decode(output['true_tokens'], skip_special_tokens=True)
            true_smiles_list.append(true_smiles)
            
            # Decode the predicted SMILES string
            pred_smiles = tokenizer.decode(output['pred_tokens'], skip_special_tokens=True)
            pred_smiles_list.append(pred_smiles)
        
        # Free memory
        self.test_step_outputs.clear()
        
        # --- Log Example Predictions ---
        print("--- Example Test Predictions ---")
        for i in range(min(5, len(true_smiles_list))):
            print(f"True     : {true_smiles_list[i]}")
            print(f"Predicted: {pred_smiles_list[i]}")
            print("-" * 20)
            
        # --- Calculate and Log Metrics ---
        exact_matches = sum(1 for true, pred in zip(true_smiles_list, pred_smiles_list) if true == pred)
        exact_match_accuracy = exact_matches / len(true_smiles_list)
        
        self.log("test_exact_match_accuracy", exact_match_accuracy)
        
        temp_dir = tempfile.mkdtemp()
        try:
            # The plotting function expects the edge matrices
            plot_results_dict = {
                "target": true_smiles_list, 
                "predictions": pred_smiles_list
            }
            # TODO : confusion matrix artifact or something
            
            # Log artifacts to MLflow
            if self.logger:
                mlflow_client = self.logger.experiment
                run_id = self.logger.run_id
                mlflow_client.log_artifacts(run_id, temp_dir, artifact_path="evaluation_plots")
                print("--- Artifacts logged successfully. ---")

        finally:
            shutil.rmtree(temp_dir)
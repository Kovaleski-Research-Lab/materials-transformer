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

from utils.eval import create_molecules_artifact

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

class SmilesTransformer(pl.LightningModule):
    def __init__(
        self,
        # arch specs
        input_dim: int,
        model_dim: int,
        num_heads: int,
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
        self.test_step_outputs = []
        
    # forward is separated into its distinct components for testing versatility
        
    def encode(self, src_spectrum):
        # Encode the IR spectrum
        src_embedded = self.embedding(src_spectrum)
        src_encoded = self.pos_encoder(src_embedded)
        return self.transformer_encoder(src_encoded)
    
    def decode(self, tgt_smiles, memory):
        # Decode the target sequence
        tgt_embedded = self.smiles_embedding(tgt_smiles)
        tgt_encoded = self.pos_encoder(tgt_embedded)
        
        # casual mask
        tgt_seq_len = tgt_smiles.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=tgt_smiles.device)
        
        decoder_output = self.transformer_decoder(
            tgt=tgt_encoded, 
            memory=memory,
            tgt_mask=tgt_mask
        )
        return self.output_fc(decoder_output)
        
    def forward(self, src_spectrum, tgt_smiles):
        memory = self.encode(src_spectrum)
        logits = self.decode(tgt_smiles, memory)
        return logits
    
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
        
        #print(f"smiles_target: {smiles_target.reshape(-1)}")
        #print(f"logits: {logits.reshape(-1, logits.shape[-1])}")

        # calc loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            smiles_target.reshape(-1),
            ignore_index = self.trainer.datamodule.tokenizer.pad_idx
        )
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
        self.log("val_loss", loss, 
                    prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def generate_smiles(self, spectrum, tokenizer, max_len=100):
        self.eval()
        
        # Encode the spectrum once
        memory = self.encode(spectrum)
        
        # Start with the <SOS> token for each item in batch
        batch_size = spectrum.shape[0]
        output_tokens = torch.full(
            (batch_size, 1),
            fill_value=tokenizer.sos_idx,
            dtype=torch.long,
            device=self.device
        )
        
        has_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for _ in range(max_len - 1):
            # decode current sequences to get logits for NEXT tokens
            logits = self.decode(output_tokens, memory)
            
            # get last predicted token for each sequence in batch
            next_tokens = logits.argmax(dim=-1)[:, -1] # shape: (batch)
            
            # Append the prediction to our output
            output_tokens = torch.cat([output_tokens, next_tokens.unsqueeze(1)], dim=1)
            
            # update tracker when sequence generates <EOS>
            has_finished = has_finished | (next_tokens == tokenizer.eos_idx)

            # Stop if we predict the <EOS> token
            if has_finished.all():
                break
                
        return output_tokens
    
    def test_step(self, batch, batch_idx):
        spectrum, smiles_true = batch

        # generate an output string
        smiles_pred = self.generate_smiles(spectrum, self.trainer.datamodule.tokenizer)
        
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
            # output['true_tokens'] is a 2D tensor of shape (batch_size, seq_len)
            for true_token_sequence in output['true_tokens']:
                true_smiles = tokenizer.decode(true_token_sequence, skip_special_tokens=True)
                true_smiles_list.append(true_smiles)
            
            for pred_token_sequence in output['pred_tokens']:
                #print(pred_token_sequence)
                pred_smiles = tokenizer.decode(pred_token_sequence, skip_special_tokens=True)
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
            create_molecules_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir)
            
            # Log artifacts to MLflow
            if self.logger:
                mlflow_client = self.logger.experiment
                run_id = self.logger.run_id
                mlflow_client.log_artifacts(run_id, temp_dir, artifact_path="evaluation_plots")
                print("--- Artifacts logged successfully. ---")

        finally:
            shutil.rmtree(temp_dir)
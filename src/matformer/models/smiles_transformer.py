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
from rdkit import Chem
from rdkit.Chem import Descriptors

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
        target_vocab_size: int,
        lambda_aux: float,
        lambda_rl: float,
        testing_method: str,
        testing_sample_num: int
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
        self.lambda_aux = lambda_aux
        self.lambda_rl = lambda_rl
        self.testing_method = testing_method
        self.testing_sample_num = testing_sample_num
        
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
        
        # ===== 4. component for Mol Weight =====
        self.mw_predictor = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
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
        return self.output_fc(decoder_output), decoder_output
        
    def forward(self, src_spectrum, tgt_smiles):
        memory = self.encode(src_spectrum)
        logits, hidden_states = self.decode(tgt_smiles, memory)
        return logits, hidden_states
    
    def configure_optimizers(self):
        optimizer = self.optimizer_cfg(params=self.parameters())
        lr_scheduler = self.scheduler_cfg(optimizer=optimizer)
        
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1,
                               "monitor": "val_loss"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def calculate_mw_for_batch(self, smiles_batch):
        """does what it sounds like"""
        mw_list = []
        for item in smiles_batch:
            smiles = self.trainer.datamodule.tokenizer.decode(item, skip_special_tokens=True)
            mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
            #print(f"smiles and its mw: {smiles} : {mw}")
            mw_list.append(mw)
        return torch.tensor(mw_list, device=smiles_batch.device)
            
    def shared_step(self, batch, batch_idx):
        spectrum, smiles_true = batch
        
        # input to decoder is the sequence minus <EOS>
        smiles_input = smiles_true[:, :-1]
        
        # target for loss is the seq minus <SOS>
        smiles_target = smiles_true[:, 1:]
        
        # get output
        logits, hidden_states = self.forward(spectrum, smiles_input)

        # calculate primary loss
        loss_ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            smiles_target.reshape(-1),
            ignore_index = self.trainer.datamodule.tokenizer.pad_idx
        )
        
        # calculate RL loss
        if self.loss_func == 'rl':
            sampled_tokens, sum_log_probs = self.get_smiles_preds(spectrum, method="sampling")
            sampled_smiles_list = []
            for smiles_sequence in sampled_tokens:
                smiles = self.trainer.datamodule.tokenizer.decode(smiles_sequence, skip_special_tokens=True)
                sampled_smiles_list.append(smiles)
                
            # calculate rewards for each string in the batch
            rewards = []
            for smiles in sampled_smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    rewards.append(1.0) # positive reward for valid SMILES
                else:
                    rewards.append(-1.0) # negative for invalid SMILES
                    
            rewards = torch.tensor(rewards, device=self.device)
            
            # for reducing variance
            baseline = rewards.mean()
            adjusted_rewards = rewards - baseline
            
            # calculate the RL loss (policy gradient)
            # want to MAXIMIZE reward, so MINIMIZE the negative of this value
            loss_rl = -torch.mean(adjusted_rewards * sum_log_probs)
            # combine
            total_loss = loss_ce + (self.lambda_rl * loss_rl)
        
        # calculate auxiliary loss
        if self.loss_func == 'aux':
            sos_hidden_state = hidden_states[:, 0, :]
            pred_mw = self.mw_predictor(sos_hidden_state).squeeze()
            true_mw = self.calculate_mw_for_batch(smiles_true)
            loss_mw = F.mse_loss(pred_mw, true_mw)
            self.log("mw_loss", loss_mw, prog_bar=True, on_step=False, on_epoch=True)
            total_loss = loss_ce + (self.lambda_aux * loss_mw)
            
        else:
            total_loss = loss_ce
            
        return total_loss
    
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
    def get_smiles_preds(self, spectrum, max_len=100, method="deterministic"):
        """This function utilizes the trained model to produce outputs and decode them into interpretable
        SMILES strings for a given batch of inputs.

        Parameters
        ----------
            spectrum (torch.tensor): The input IR spectrum samples passed to the model.
            max_len (int, optional): maximum SMILES string length to predict. Defaults to 100.
            method (str, optional): prediction strategy. Defaults to "deterministic", alternative is "sampling"

        Returns
        -------
            output_tokens (torch.tensor): (batch_size, num_preds) SMILES string predictions
        """
        self.eval()
        
        # Encode the spectrum once
        memory = self.encode(spectrum)
        
        # Start with the <SOS> token for each item in batch
        tokenizer = self.trainer.datamodule.tokenizer
        batch_size = spectrum.shape[0]
        output_tokens = torch.full(
            (batch_size, 1),
            fill_value=tokenizer.sos_idx,
            dtype=torch.long,
            device=self.device
        )
        # log probabilities of actions taken
        sum_log_probs = torch.zeros(batch_size, device=self.device)
        
        has_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for _ in range(max_len - 1):
            # decode current sequences to get logits for NEXT tokens
            logits, _ = self.decode(output_tokens, memory)
            
            if method == "deterministic":
                # for the next position, get the most likely token
                next_tokens = logits.argmax(dim=-1)[:, -1] # shape: (batch)
                # Append the prediction to our output
                output_tokens = torch.cat([output_tokens, next_tokens.unsqueeze(1)], dim=1)
                
            elif method == "sampling": # sample from a distr instead of just taking the best
                next_token_logits = logits[:, -1, :] # logits for very last token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1) # sample
                # get log prob of the just-sampled token
                log_prob = F.log_softmax(logits[:, -1, :], dim=-1)
                sum_log_probs += log_prob.gather(1, next_tokens).squeeze()
                
                # Append the prediction to our output
                output_tokens = torch.cat([output_tokens, next_tokens], dim=1)
            
            # update tracker when sequence generates <EOS>
            has_finished = has_finished | (next_tokens == tokenizer.eos_idx)

            # Stop if we predict the <EOS> token
            if has_finished.all():
                break
                
        return output_tokens, sum_log_probs
    
    def test_step(self, batch, batch_idx):
        spectrum, smiles_true = batch

        if self.testing_method == "determinsitic": # getting the most likely output
            smiles_pred, _ = self.get_smiles_preds(spectrum, method=self.testing_method)
            smiles_pred = smiles_pred.detach().cpu()
            
        elif self.testing_method == "sampling": # sampling a range of outputs
            smiles_pred = []
            for _ in range(self.testing_sample_num):
                smiles_pred_item, _ = self.get_smiles_preds(spectrum, method=self.testing_method)
                smiles_pred_item = smiles_pred_item.detach().cpu()
                smiles_pred.append(smiles_pred_item)
                
        output = {
            "pred_tokens": smiles_pred,
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
            
            if self.testing_method == "deterministic": # just a single one to decode per
                for pred_token_sequence in output['pred_tokens']:
                    #print(pred_token_sequence)
                    pred_smiles = tokenizer.decode(pred_token_sequence, skip_special_tokens=True)
                    pred_smiles_list.append(pred_smiles)
            elif self.testing_method == "sampling": # we have multiple preds per
                for pred_strings_list in output['pred_tokens']:
                    preds = []
                    for pred_token_sequence in pred_strings_list:
                        pred_smiles = tokenizer.decode(pred_token_sequence, skip_special_tokens=True)
                        preds.append(pred_smiles)
                    pred_smiles_list.append(preds)
                        
        # Free memory
        self.test_step_outputs.clear()
        
        # --- Log Example Predictions ---
        print("--- Example Test Predictions ---")
        for i in range(min(5, len(true_smiles_list))):
            print(f"True     : {true_smiles_list[i]}")
            if self.testing_method == "deterministic":
                print(f"Predicted: {pred_smiles_list[i]}")
            elif self.testing_method == "sampling":
                print(f"Predicted: {pred_smiles_list[i][0]}")
            print("-" * 20)
            
        # --- Calculate and Log Metrics ---
        if self.testing_method == "deterministic":
            exact_matches = sum(1 for true, pred in zip(true_smiles_list, pred_smiles_list) if true == pred)
            exact_match_accuracy = exact_matches / len(true_smiles_list)
            self.log("test_top_1_accuracy", exact_match_accuracy)
        elif self.testing_method == "sampling":
            top_1_matches = 0
            top_5_matches = 0
            top_10_matches = 0

            for true_smiles, pred_list in zip(true_smiles_list, pred_smiles_list):
                # check for Top-1 match
                if true_smiles == pred_list[0]:
                    top_1_matches += 1
                    top_5_matches += 1  # a top-1 match is also a top-5 and top-10 match
                    top_10_matches += 1
                    continue # move to the next true_smiles

                # check for Top-5 match
                if true_smiles in pred_list[0:5]:
                    top_5_matches += 1
                    top_10_matches += 1 # a top-5 match is also a top-10 match
                    continue

                # check for Top-10 match (last chance! goooo)
                if true_smiles in pred_list:
                    top_10_matches += 1
            
            # calculate and log accuracies
            total_count = len(true_smiles_list)
            self.log("test_top_1_accuracy", top_1_matches / total_count)
            self.log("test_top_5_accuracy", top_5_matches / total_count)
            self.log("test_top_10_accuracy", top_10_matches / total_count)
        
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
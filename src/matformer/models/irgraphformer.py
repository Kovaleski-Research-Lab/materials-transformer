import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch_geometric.nn import TransformerConv
from typing import Any
import tempfile
import shutil
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
        # graph specs
        num_atom_types: int
    ):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.depth = depth
        self.num_heads = num_heads
        self.rows = rows
        self.cols = cols
        self.dropout = dropout
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = lr_scheduler
        self.num_atom_types = num_atom_types
        self.loss_func = loss_func
        self.seq_len = seq_len # IR spectrum length
        
        super().__init__()
        self.save_hyperparameters()
        
        # ===== 1. Sequence Encoder (IR Spectrum) =====
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, batch_first=True)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=depth)
        
        # ===== 2. Graph Decoder =====
        self.initial_node_embeddings = nn.Embedding(80, model_dim)
        
        # stack of Graph Transformer layers to refine the graph
        self.graph_transformer_layers = nn.ModuleList([
            TransformerConv(in_channels=model_dim, out_channels=model_dim // self.num_heads, 
                            heads=self.num_heads, dropout=self.dropout)
            for _ in range(depth)
        ])
        
        # ===== 3. Output Heads =====
        self.node_predictor = nn.Linear(model_dim, 80)
        
        # Predicts the existence of a bond between any two atoms (edge)
        self.edge_predictor = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1)
        )
        
        # store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_step_outputs = []
        
    def forward(self, ir_spectrum):
        '''x = self.embedding(x) * math.sqrt(self.model_dim)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x) # (batch, seq_len, model_dim)
        x = output.permute(0, 2, 1) # (batch, model_dim, seq_len)
        x = self.conv_decoder(x)
        x = x.flatten(start_dim=1)
        logits = self.final_decoder(x)
        #flattened_output = output.flatten(start_dim=1)
        #projected = self.output_decoder(flattened_output)
        #out = torch.sigmoid(projected)
        return logits.view(-1, self.rows, self.cols)'''
        # 1. Encode the IR spectrum into a single context vector
        spec_embedded = self.embedding(ir_spectrum) * math.sqrt(self.hparams.model_dim)
        spec_encoded = self.pos_encoder(spec_embedded)
        spec_context = self.transformer_encoder(spec_encoded).mean(dim=1) # (batch, model_dim)

        # 2. Initialize the graph nodes
        batch_size = ir_spectrum.shape[0]
        node_indices = torch.arange(80, device=self.device).expand(batch_size, -1)
        node_features = self.initial_node_embeddings(node_indices) # (batch, max_atoms, model_dim)
        
        # Condition the initial nodes with the spectrum's context
        node_features = node_features + spec_context.unsqueeze(1)

        # 3. Iteratively refine the graph structure
        # We need a placeholder fully-connected edge_index for the graph transformer
        adj_matrix = torch.ones(80, 80, device=self.device)
        edge_index = adj_matrix.to_sparse().indices()
        
        for layer in self.graph_transformer_layers:
            # TransformerConv expects (num_nodes, features), so we must reshape
            num_nodes_total = batch_size * 80
            node_features = node_features.view(num_nodes_total, self.hparams.model_dim)
            
            # Since each graph in the batch is disconnected, we process them together
            node_features = layer(node_features, edge_index)
            
            # Reshape back to batch format
            node_features = node_features.view(batch_size, 80, -1)

        # 4. Predict atom types (nodes) and bonds (edges)
        predicted_node_types = self.node_predictor(node_features) # (batch, max_atoms, num_atom_types)

        # To predict edges, check every pair of nodes
        n = 80
        ii, jj = torch.triu_indices(n, n, offset=1) # Get upper-triangular indices
        
        # Get embeddings for each pair
        node_pairs = torch.cat([
            node_features[:, ii, :],
            node_features[:, jj, :]
        ], dim=-1) # (batch, num_pairs, model_dim * 2)

        predicted_edges = self.edge_predictor(node_pairs).squeeze(-1) # (batch, num_pairs)

        # Reconstruct the adjacency matrix from the pair predictions
        adj = torch.zeros(batch_size, n, n, device=self.device, dtype=predicted_edges.dtype)
        adj[:, ii, jj] = predicted_edges
        adj = adj + adj.transpose(1, 2) # Make symmetric

        return predicted_node_types, adj
    
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
        samples, labels = batch
        true_atom_types, true_adj_matrix = labels
        
        pred_atom_types, pred_adj_matrix = self.forward(samples)

        # --- Combined Loss Calculation ---
        # 1. Loss for predicting atom types (Nodes)
        # Note: permute is needed as CrossEntropyLoss expects (N, C, ...)
        loss_nodes = F.cross_entropy(
            pred_atom_types.permute(0, 2, 1), 
            true_atom_types
        )
        
        # 2. Loss for predicting bonds (Edges)
        loss_edges = F.binary_cross_entropy_with_logits(
            pred_adj_matrix, 
            true_adj_matrix
        )

        # Combine the losses
        total_loss = loss_nodes + loss_edges
        
        # Return the total loss and a dictionary of all components for logging
        loss_dict = {
            "loss": total_loss, 
            "loss_nodes": loss_nodes, 
            "loss_edges": loss_edges
        }
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch, batch_idx)
        # Log the metrics
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, 
                    prog_bar=True, on_step=False, on_epoch=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch, batch_idx)
        # Log the metrics
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, 
                    on_step=False, on_epoch=True)
        return loss_dict["loss"]
    
    def test_step(self, batch, batch_idx):
        samples, (true_atom_types, true_adj_matrix) = batch
        pred_atom_types, pred_adj_matrix = self.forward(samples)

        # store all components needed for aggregation at the end of the epoch
        output = {
            "pred_nodes": pred_atom_types.detach().cpu(),
            "pred_edges": pred_adj_matrix.detach().cpu(),
            "true_nodes": true_atom_types.detach().cpu(),
            "true_edges": true_adj_matrix.detach().cpu()
        }
        self.test_step_outputs.append(output)
        
        return output
    
    '''def test_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch, batch_idx)
        samples, labels = batch
        self.log("test_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        
        # keep preds and labels for aggregation
        output = {"preds": preds.detach().cpu().numpy(), "labels": labels.detach().cpu().numpy()}
        self.test_step_outputs.append(output)
        return output'''
        
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("Test step outputs were empty. Skipping artifact generation.")
            return

        # Aggregate each component from the dictionaries saved in test_step
        all_pred_nodes = torch.cat([x['pred_nodes'] for x in self.test_step_outputs])
        all_pred_edges = torch.cat([x['pred_edges'] for x in self.test_step_outputs])
        all_true_nodes = torch.cat([x['true_nodes'] for x in self.test_step_outputs])
        all_true_edges = torch.cat([x['true_edges'] for x in self.test_step_outputs])
        
        # Free memory
        self.test_step_outputs.clear()
        
        # --- calculate and log node accuracy ---
        # Convert node prediction logits to class predictions
        pred_node_labels = torch.argmax(all_pred_nodes, dim=-1)
        # Create a mask to exclude padding tokens from the accuracy calculation
        non_pad_mask = (all_true_nodes != ATOM_MAP['PAD'])
        node_accuracy = (pred_node_labels[non_pad_mask] == all_true_nodes[non_pad_mask]).float().mean()
        self.log("test_node_accuracy", node_accuracy, sync_dist=True)
        
        temp_dir = tempfile.mkdtemp()
        try:
            # The plotting function expects the edge matrices
            plot_results_dict = {
                "target": all_true_edges.numpy(), 
                "predictions": all_pred_edges.numpy()
            }
            create_matrix_artifact(eval_df=plot_results_dict, artifacts_dir=temp_dir)
            
            # Log artifacts to MLflow
            if self.logger:
                mlflow_client = self.logger.experiment
                run_id = self.logger.run_id
                mlflow_client.log_artifacts(run_id, temp_dir, artifact_path="evaluation_plots")
                print("--- Artifacts logged successfully. ---")

        finally:
            shutil.rmtree(temp_dir)
        
        
        
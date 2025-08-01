# OLD, UPDATE FOR SMILES



#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import itertools
import os
import sys
import torch
import logging
import numpy as np
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import pickle
import torch
from tqdm import tqdm
from typing import Dict
import pandas as pd
from rdkit import Chem

def graph_collate_fn(batch):
    """
    Takes a list of samples from TensorDataset and groups the labels.
    """
    # Unzip the batch: [(X1, y_atom1, y_adj1), (X2, y_atom2, y_adj2), ...]
    spectra, atom_labels, adj_labels = zip(*batch)
    
    # Stack samples and labels into batch tensors
    spectra_batch = torch.stack(spectra, 0)
    atom_labels_batch = torch.stack(atom_labels, 0)
    adj_labels_batch = torch.stack(adj_labels, 0)
    
    # Return in the format the model expects
    return spectra_batch, (atom_labels_batch, adj_labels_batch)

# ----------------------
# DATAMODULE and DATASET
# ----------------------

ATOM_MAP = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'S': 4, 'Cl': 5, 'PAD': 6}

class SmilesDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_cpus: int,
        data_path: str,
        smiles_column: str,
        max_smiles_len: int, # for padding
        val_split: float = 0.2
    ):
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.data_path = data_path
        self.val_split = val_split
        self.smiles_column = smiles_column
        self.max_smiles_len = max_smiles_len
        
        # placeholders
        self.char_to_idx = None
        self.idx_to_char = None
        self.pad_token_idx = None
        self.vocab_size = None
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        
        super().__init__()
    
    def setup(self, stage):
        # read in the parquet file
        df = pd.read_parquet(self.data_path)

        # build the vocab - collect all unique chars
        all_smiles = df[self.smiles_column].tolist()
        char_set = set()
        for smiles in all_smiles:
            char_set.update(list(smiles))
            
        # define special tokens and create the vocab
        special_tokens = ['<PAD>', '<SOS>', '<EOS>']
        vocab = sorted(list(char_set)) + special_tokens
        
        # create mappings
        self.idx_to_char = {i: char for i, char in enumerate(vocab)}
        self.char_to_idx = {char: i for i, char in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        self.pad_token_idx = self.char_to_idx['<PAD>']
        
        # process IR spectra
        X = np.log(np.vstack(df['y'].values) + 1)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        
        # tokenize and pad smiles strings
        tokenized_smiles = []
        sos_token = self.char_to_idx['<SOS>']
        eos_token = self.char_to_idx['<EOS>']
        
        for smiles in all_smiles:
            tokens = [self.char_to_idx[char] for char in smiles]
            tokenized = [sos_token] + tokens + [eos_token]
            
            # pad to max length
            padding_needed = self.max_smiles_len - len(tokenized)
            padded_tokens = tokenized + [self.pad_token_idx] * padding_needed
            tokenized_smiles.append(padded_tokens[:self.max_smiles_len])
            
        y_tensor = torch.tensor(tokenized_smiles, dtype=torch.long)
        
        # create final dataset and splits
        full_dataset = TensorDataset(X_tensor, y_tensor)
        
        dataset_size = len(full_dataset)
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size
        
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.test_dataset = self.val_dataset
        
    # DATALOADERS
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_cpus,
            shuffle=True,
            persistent_workers=True,
            collate_fn=graph_collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_cpus,
            shuffle=False,
            persistent_workers=True,
            collate_fn=graph_collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_cpus,
            shuffle=False,
            persistent_workers=True,
            collate_fn=graph_collate_fn
        )
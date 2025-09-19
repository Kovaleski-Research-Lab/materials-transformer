#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd

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
        ir_column: str,
        max_smiles_len: int, # for padding
        val_split: float = 0.2
    ):
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.data_path = data_path
        self.val_split = val_split
        self.smiles_column = smiles_column
        self.ir_column = ir_column
        self.max_smiles_len = max_smiles_len
        
        # placeholders
        self.char_to_idx = None
        self.idx_to_char = None
        self.pad_token_idx = None
        self.vocab_size = None
        self.tokenizer = None
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.mw_dict = {}
        
        super().__init__()
    
    def setup(self, stage):
        # read in the parquet file
        df = pd.read_parquet(self.data_path)

        # create and store the tokenizer
        all_smiles = df[self.smiles_column].tolist()
        self.tokenizer = SMILESTokenizer.from_smiles_list(all_smiles)
        
        # process IR spectra
        X = np.log(np.vstack(df[self.ir_column].values) + 1)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        
        # tokenize and pad smiles strings
        tokenized_smiles = []
        for smiles in all_smiles:
            tokenized = self.tokenizer.encode(smiles)
            
            # pad to max length
            padding_needed = self.max_smiles_len - len(tokenized)
            padded_tokens = tokenized + [self.tokenizer.pad_idx] * padding_needed
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
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_cpus,
            shuffle=False,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_cpus,
            shuffle=False,
            persistent_workers=True
        )
        
# ----------------------
# TOKENIZER
# ----------------------

class SMILESTokenizer:
    def __init__(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        
        # special indices
        self.pad_idx = char_to_idx['<PAD>']
        self.sos_idx = char_to_idx['<SOS>']
        self.eos_idx = char_to_idx['<EOS>']
        self.vocab_size = len(char_to_idx)
        
    @classmethod 
    def from_smiles_list(cls, smiles_list):
        """Factory method to create a tokenizer from a list of SMILES strings."""
        char_set = set()
        for smiles in smiles_list:
            char_set.update(list(smiles))

        special_tokens = ['<PAD>', '<SOS>', '<EOS>']
        vocab = special_tokens + sorted(list(char_set))
        
        idx_to_char = {i: char for i, char in enumerate(vocab)}
        char_to_idx = {char: i for i, char in enumerate(vocab)}
        
        return cls(char_to_idx, idx_to_char)
        
    def encode(self, smiles_string):
        """Converts SMILES string to a list of integer tokens"""
        tokens = [self.char_to_idx[char] for char in smiles_string]
        return [self.sos_idx] + tokens + [self.eos_idx]
        
    def decode(self, token_list, skip_special_tokens=True):
        """Converts list of integer tokens back to SMILES string"""
        token_list = token_list.flatten()
        chars = []
        for token_tensor in token_list:
            token = token_tensor.item()
            if token == self.eos_idx and skip_special_tokens:
                break
            if skip_special_tokens and token in [self.pad_idx, self.sos_idx]:
                continue
            chars.append(self.idx_to_char[token])
        return "".join(chars)
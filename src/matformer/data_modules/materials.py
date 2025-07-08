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
from torch.utils.data import Dataset, DataLoader, Subset
import pickle
import torch
from tqdm import tqdm
from typing import Dict
import pandas as pd

# ----------------------
# DATAMODULE and DATASET
# ----------------------

class IRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_cpus: int,
        data_path: str
    ):
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.data_path = data_path
    
    def setup(self, stage):
        # read in the parquet file
        raw_data = pd.read_parquet(self.data_path)
        
    
class IR_Dataset(Dataset):
    """
    Dataset for the IR data (prob will need to change this)
    """
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)
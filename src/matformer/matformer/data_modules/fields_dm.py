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

# ----------------------
# DATAMODULE and DATASET
# ----------------------

class NFDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_cpus: int,
        near_field_dim: int,
        seq_len: int,
        data_path: str,
        spacing_mode: str,
        io_mode: str
    ):
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.near_field_dim = near_field_dim
        self.seq_len = seq_len
        self.data_path = data_path
        self.spacing_mode = spacing_mode
        self.io_mode = io_mode
        self.index_map = None
        self.dataset = None
        self.train = None
        self.valid = None
        self.test = None
        
        super().__init__()
        
    def setup(self, stage):
        # read the data file and set the constant split
        raw_data = torch.load(self.data_path, weights_only=True)
        self.index_map = self._create_index_map(raw_data)
        
        # process the data
        self.dataset = self.preprocess_data(raw_data)
        
        # init train val
        self.train = Subset(self.dataset, self.index_map['train'])
        self.valid = Subset(self.dataset, self.index_map['valid'])
        # test is valid (for now, will add functionality for tweaking this later)
        self.test = Subset(self.dataset, self.index_map['valid'])
        
        
    def _create_index_map(self, data):
        index_map = {'train': [], 'valid': []}
        for i in range(len(data['tag'])):
            key = 'valid' if data['tag'][i] == 0 else 'train'
            index_map[key].append(i)
        return index_map
    
    # DATALOADERS
    
    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          shuffle=True,
                          persistent_workers=True)
        
    def val_dataloader(self):
        return DataLoader(self.valid,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          shuffle=False,
                          persistent_workers=True)
        
    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          shuffle=False,
                          persistent_workers=True)
    
    def preprocess_data(self, raw_data, order=(-1, 0, 1, 2)):
        """Formats the preprocessed data file into the correct setup  
        and order for the temporal model.

        Args:
            order (tuple, optional): order of the sequence to be used. Defaults to (-1, 0, 1, 2).
            
        Returns:
            dataset (NF_Dataset): formatted dataset
        """
        
        all_samples, all_labels = [], []
        
        fields = raw_data['near_fields']
        
        #if stage == 'train': # normalize
        #fields = mapping.l2_norm(data['near_fields'])
            
        # [samples, 2, xdim, ydim, 63] --> access each of the datapoints
        for i in range(fields.shape[0]):       
            full_sequence = fields[i] # [2, xdim, ydim, total_slices]

            total = full_sequence.shape[-1] # all time slices
            
            if self.spacing_mode == 'distributed':
                if self.io_mode == 'one_to_many':
                    # calculate seq_len+1 evenly spaced indices
                    indices = np.linspace(1, total-1, self.seq_len+1)
                    distributed_block = full_sequence[:, :, :, indices]
                    # the sample is the first one, labels are the rest
                    sample = distributed_block[:, :, :, :1]  # [2, xdim, ydim, 1]
                    label = distributed_block[:, :, :, 1:]  # [2, xdim, ydim, seq_len]
                    
                elif self.io_mode == 'many_to_many':
                    # Calculate seq_len+1 evenly spaced indices for input and shifted output
                    indices = np.linspace(0, total-1, self.seq_len+1).astype(int)
                    distributed_block = full_sequence[:, :, :, indices]
                    
                    # Input sequence: all but last timestep
                    sample = distributed_block[:, :, :, :-1]  # [2, xdim, ydim, seq_len]
                    # Output sequence: all but first timestep
                    label = distributed_block[:, :, :, 1:]   # [2, xdim, ydim, seq_len]
                    
                else:
                    # many to one, one to one not implemented
                    raise NotImplementedError(f'Specified recurrent input-output mode is not implemented.')
                
                # rearrange dims and add to lists
                sample = sample.permute(order) # [1, 2, xdim, ydim]
                label = label.permute(order) # [seq_len, 2, xdim, ydim]
                all_samples.append(sample)
                all_labels.append(label)
                
            elif self.spacing_mode == 'sequential':
                if self.io_mode == 'one_to_many':
                    #for t in range(0, total, conf['seq_len']+1): note: this raise the total number of sample/label pairs
                    t = 0
                    # check if there are enough timesteps for a full block
                    if t + self.seq_len < total:
                        block = full_sequence[:, :, :, t:t+self.seq_len + 1]
                        # ex: sample -> t=0 , label -> t=1, t=2, t=3 (if seq_len were 3)
                        sample = block[:, :, :, 0:1]
                        label = block[:, :, :, 1:]
                            
                elif self.io_mode == 'many_to_many':
                    # true many to many
                    sample = full_sequence[:, :, :, :self.seq_len]
                    label = full_sequence[:, :, :, 1:self.seq_len+1]
                    
                else:
                    raise NotImplementedError(f'Specified recurrent input-output mode is not implemented.')
                    
                sample = sample.permute(order)
                label = label.permute(order)
                all_samples.append(sample)
                all_labels.append(label)
            
            else:
                # no other spacing modes are implemented
                raise NotImplementedError(f'Specified recurrent dataloading confuration is not implemented.')
            
        return NF_Dataset(all_samples, all_labels)
    
class NF_Dataset(Dataset):
    """
    Dataset for the sequential transformer
    """
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)
    

        
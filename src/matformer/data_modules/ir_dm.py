#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
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

class IRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_cpus: int,
        data_path: str,
        val_split: float = 0.2
    ):
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.data_path = data_path
        self.val_split = val_split
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        
        super().__init__()
    
    def setup(self, stage):
        # read in the parquet file
        raw_df = pd.read_parquet(self.data_path)
        # reconstruct OG 2D matrices
        raw_df['Merged Adjacency Matrix'] = raw_df.apply(
            lambda row: np.array(row['Merged Adjacency Matrix']).reshape(row['Matrix_Shape']),
            axis=1
        )
        # process LABELS - adjacency matrices
        y = np.stack(raw_df['Merged Adjacency Matrix'].values) # (num_samples, 80, 160)
        # process FEATURES - IR spectra
        X = np.log(np.vstack(raw_df['y'].values + 1))
        
        # convert to tensors - transformer expects a specific feature dimensionality
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # (num_samples, seq_len, 1)
        
        y_atoms_padded = []
        y_adj_padded = []
        pad_token = ATOM_MAP['PAD']

        # process SMILES to get Node and Edge Labels
        for smiles in raw_df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue # Skip if RDKit can't parse the SMILES

            # Get Atom Types and pad them
            atom_labels = [ATOM_MAP.get(atom.GetSymbol(), pad_token) for atom in mol.GetAtoms()]
            padding_needed = 80 - len(atom_labels)
            padded_atom_labels = atom_labels + [pad_token] * padding_needed
            y_atoms_padded.append(padded_atom_labels)

            # Get Adjacency Matrix and pad it
            adj_matrix = Chem.GetAdjacencyMatrix(mol)
            num_actual_atoms = adj_matrix.shape[0]
            padded_adj = np.zeros((80, 80))
            padded_adj[:num_actual_atoms, :num_actual_atoms] = adj_matrix
            y_adj_padded.append(padded_adj)
        
        # 3. Convert labels to Tensors
        y_atoms_tensor = torch.tensor(y_atoms_padded, dtype=torch.long)
        y_adj_tensor = torch.tensor(np.array(y_adj_padded), dtype=torch.float32)
        
        #y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 80, 160)
        #print(f"\nX_tensor size: {X_tensor.shape}\ny_tensor size: {y_tensor.shape}")
        
        # create dataset
        full_dataset = TensorDataset(X_tensor, y_atoms_tensor, y_adj_tensor)
        
        # create train/val sets
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
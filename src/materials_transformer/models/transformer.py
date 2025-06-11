import pytorch_lightning as pl
from omegaconf import DictConfig

class MaterialsTransformerModel(pl.LightningModule):
    def __init__(
        self,
        # architecture specifics
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        # relevant hyperparameters 
        learning_rate: float,
        scheduler_cfg: DictConfig,
        near_field_dim: int,
        loss_func: str,
        seq_len: int
    ):  
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.scheduler_cfg = scheduler_cfg
        self.near_field_dim = near_field_dim
        self.loss_func = loss_func
        self.seq_len = seq_len
        
        self.t_in_for_pos_embed = 1 # num of input steps pos embed should handle
        self.max_steps = self.t_in_for_pos_embed + self.seq_len
        
        super().__init__()
        self.save_hyperparameters()
        
from encoder.mamba import MambaEncoder
from head.pred_embed import PredEmbed
from utils import BaseJepaPredictor
from jepa import JEPA
from etl.obj import IndexData
import torch

from utils import train_jepa_model
from torch.utils.data import DataLoader

def train_mamba_encoder_grid(params):
    # Initialize the MambaEncoder
    mamba_encoder = MambaEncoder(
        seq_len=params['seq_len'], 
        embed_dim=params['mamba_encoder_embed_dim']
    )

    if 'mamba_encoder_pth_path' in params.keys():
        mamba_encoder = MambaEncoder(
            seq_len=params['seq_len'], 
            embed_dim=params['mamba_encoder_embed_dim']
        )
        mamba_encoder.load_state_dict(torch.load(params['mamba_encoder_pth_path']))

    jepa_model = JEPA(
        encoder=mamba_encoder, 
        predictor=BaseJepaPredictor(
            input_size=params['input_size'], 
            mask_ratio=params['mask_ratio']
        )
    )

    if 'jepa_model_pth_path' in params.keys():
        mamba_encoder = MambaEncoder(
            seq_len=params['seq_len'], 
            embed_dim=params['mamba_encoder_embed_dim']
        )
        mamba_encoder.load_state_dict(torch.load(params['jepa_model_pth_path']))


    train_dataset = IndexData(
        df=params['train_df'],
        col='Open',
        indexes='all',
        patch_length=params['patch_length'],
        stride=params['train_stride'],
        return_y=False
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params['batch_size'], 
        shuffle=True
    )

    val_dataset = IndexData(
        df=params['val_df'],
        col='Open',
        indexes='all',
        patch_length=params['patch_length'],
        stride=params['val_stride'],
        return_y=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params['batch_size'], 
        shuffle=True
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    jepa_model, train_losses, val_losses = train_jepa_model(
        model=jepa_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=params['num_epochs'],
        early_stopping_patience=params['early_stopping_patience'],
        model_name=params['model_name'],
        device=device
    )
    return jepa_model
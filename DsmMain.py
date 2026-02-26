import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor
import os
from pytorch_lightning.loggers import WandbLogger
import wandb

from DsmNet import SirenMRI
from MRIDatasets import MRI4Ddataset, MRI4Ddataset_Val

#set seed

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('medium')
torch.manual_seed(seed=0)
torch.cuda.manual_seed(seed=0)
torch.mps.manual_seed(seed=0)

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def main_imp():
    device = torch.device("mps")#"cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    mri_4d = torch.load('paediatric_mri/atrt_00001/t0/mri_seq.pt', weights_only=False)
    seq = mri_4d.shape[0]
    brain_mask = (mri_4d.abs().sum(dim=0) > 10000)
    no_brain_mask = (mri_4d.abs().sum(dim=0) <= 10000)
    means = []
    stds  = []
    for c in range(seq):
        vals = mri_4d[c][brain_mask]
        mean = vals.mean()
        std  = vals.std(unbiased=False)
        means.append(mean)
        stds.append(std)

    means = torch.stack(means)
    stds  = torch.stack(stds)
    stds = torch.clamp(stds, min=1e-6)
    mri_4d_norm = (mri_4d - means[:, None, None, None]) / stds[:, None, None, None]
    mri_4d_norm = torch.clamp(mri_4d_norm, -10.0, 10.0)
    mri_4d_norm[:,no_brain_mask] = -2.0

    train_data = MRI4Ddataset(mri_4d_norm, sample_size=128*128)
    print("len train: ", len(train_data))
    val_data = MRI4Ddataset_Val(mri_4d_norm[:,12:13,:,:]) #to validate on 2D slice

    #Dataloader
    train_loader = DataLoader(train_data, batch_size=128, pin_memory=True, num_workers=10, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data), pin_memory=False, num_workers=0)
    #Logger
    wandb_logger = WandbLogger(name="Enfold", project="MRI-manifold")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    #create model instance
    mapping_size = 256
    B_gauss = torch.randn((mapping_size, seq)).to(device)
    model =SirenMRI(in_features=mapping_size*2, B=B_gauss*0.1, out_features=1,
                hidden_layers=3)

    # Train the model
    trainer = pl.Trainer(max_epochs=101,
                        # accelerator="cpu",
                        # devices=2,
                        logger=wandb_logger,
                        callbacks=[lr_monitor],
                        check_val_every_n_epoch=10,
                        gradient_clip_val=10,
                        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    torch.save(model, 'model-mri-manifold.pt')

if __name__ == "__main__":
    main_imp()
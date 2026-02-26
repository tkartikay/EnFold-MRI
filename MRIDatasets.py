import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device("mps")#"cuda" if torch.cuda.is_available() else "cpu")
    
class MRI4Ddataset(Dataset):
    def __init__(self, image_tensor, sample_size=2):
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        self.idx = 0
        """
        Args:
            image_tensor (torch.Tensor): A 4D tensor with multisequence MRI images seq x d x h x w.
        """

        # image_tensor = torch.load('4Dimage.pt', weights_only=False) #weightsonly default changed pytorch=2.6

        # Generate coordinates
        T, D, H, W = image_tensor.shape
        print("Image Tensor Shape: ", image_tensor.shape)
        print("N sequences : ", T)
        self.coords = torch.stack(torch.meshgrid(torch.arange(D),
                                  torch.arange(H), torch.arange(W)),
                                  -1).reshape(-1, 3).float()
        self.intensities = image_tensor
        self.sample_size = sample_size

        # to keep only inside patient points
        sum_over_t = image_tensor.sum(dim=0)  # [D, H, W]
        fg_mask_3d = sum_over_t > -2.0      # choose suitable threshold

        fg_mask = fg_mask_3d.reshape(-1)      # [N]
        self.valid_indices = torch.nonzero(fg_mask, as_tuple=False).squeeze(1) #N_valid


    def __len__(self):
        return self.sample_size

    def __getitem__(self, _):
        rand_pos = torch.randint(0, self.valid_indices.shape[0], (1,)).item()
        idx = self.valid_indices[rand_pos]

        # Fetch the intensity values at the selected flattened coordinate idx across all sequences
        intensities = self.intensities.reshape(self.intensities.shape[0], -1)[:, idx].reshape(-1)
        return intensities
    
class MRI4Ddataset_Val(Dataset):
    def __init__(self, img):
        torch.manual_seed(0)
        np.random.seed(seed=0)
        # Generate coordinates
        N, D, H, W = img.shape
        self.coords = torch.stack(torch.meshgrid(torch.arange(D),
                                  torch.arange(H), torch.arange(W)),
                                  -1)
        self.coords = self.coords[:,:,:].reshape(-1, 3).float()
        self.intensities = img


    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, index):
        intensities = self.intensities.reshape(self.intensities.shape[0], -1)[:, index].reshape(-1)
        return intensities
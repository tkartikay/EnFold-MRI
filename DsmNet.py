from typing import Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics
from pathlib import Path

device = torch.device("mps")#"cuda" if torch.cuda.is_available() else "cpu")

##https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        # self.bn = nn.BatchNorm1d(num_features=out_features)

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class SirenMRI(pl.LightningModule):
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=7, latent_features=4, out_features=1,
                 B=None, Bx=None, Bt=None, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.B = B
        latent_features_b = 2
        self.numfreq = 256
        self.once = 1
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))


        self.net = nn.Sequential(*self.net)
        self.k_sine = SineLayer(hidden_features, out_features,
                                is_first=False, omega_0=hidden_omega_0)
        self.k_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            self.k_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                           np.sqrt(6 / hidden_features) / hidden_omega_0)


    def forward(self, x):
        if self.B is not None:
            x = torch.matmul(2. * torch.pi * x, self.B.T).to(device)
            x = torch.cat([torch.sin(x), torch.cos(x)], -1).to(device)

        xf = self.net(x)
        ki = self.k_linear(xf)

        return ki
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-5, amsgrad=True)
        sch1 = torch.optim.lr_scheduler.ConstantLR(optim, factor=1, total_iters=200)
        sch2 = torch.optim.lr_scheduler.ConstantLR(optim, factor=0.1, total_iters=3000)
        sch = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[sch1, sch2], milestones=[200])
        return ([optim], [sch])
    
    
    def training_step(self, batch, batch_idx):
        u = batch

        if self.current_epoch == 0 and batch_idx == 0:
            print("Batch 0 inputs: ", u.shape)
        
        # log-uniform sigma per sample
        sigma_min, sigma_max = 0.1, 1
        sigmas = torch.exp(
            torch.empty(u.shape[0], device=device).uniform_(np.log(sigma_min), np.log(sigma_max))
        )
        sigmas = sigmas[:, None]  # (B,1) broadcast to (B,6)
        loss = dsm_energy_loss(self, u, sigma=sigmas)
        self.log('train dsm loss', loss)
        return loss
    
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     if batch_idx == 0:
    #         g = self.k_linear.weight.grad
    #         print("grad exists:", g is not None, "grad norm:", None if g is None else g.norm().item())
    #     return

    def validation_step(self, batch, batch_idx):
        u = batch

        if self.current_epoch == 0 and batch_idx == 0:
            print("Batch 0 val: ", u.shape)

        h, w = 334,441
        gt_img = u[:,1].view(h,w)
        # Energy image
        with torch.no_grad():
            E = self(u).squeeze()  # (N,)
        E_img = E.view(h, w)

        # Gradient norm (needs grad)
        with torch.enable_grad():
            u_in = u.detach().clone()
            u_in.requires_grad_(True)
            E_in = self(u_in)
            if E_in.ndim == 2 and E_in.shape[1] == 1:
                E_sum = E_in.sum()
            else:
                E_sum = E_in.view(-1).sum()
            grad = torch.autograd.grad(E_sum, u_in, create_graph=True)[0]  # (N,6)
            Gn_img = grad.norm(dim=1).view(h, w)

            # Hessian trace (Laplacian) via diagonal (moderate cost)
            hess_diag = []
            for i in range(u_in.shape[1]):
                second = torch.autograd.grad(grad[:, i].sum(), u_in, retain_graph=True)[0][:, i]
                hess_diag.append(second)
            hess_diag = torch.stack(hess_diag, dim=1)  # (N,6)
            Hs_img = hess_diag.sum(dim=1).view(h, w)

        fig0 = plt.figure()
        plt.imshow(gt_img.detach().cpu().numpy(), cmap='binary_r')
        plt.colorbar()
        fig1 = plt.figure()
        plt.imshow(E_img.detach().cpu().numpy(), cmap='jet')
        plt.colorbar()
        fig2 = plt.figure()
        plt.imshow(Gn_img.detach().cpu().numpy(), cmap='jet')
        plt.colorbar()
        fig3 = plt.figure()
        plt.imshow(Hs_img.detach().cpu().numpy(), cmap='jet')
        plt.colorbar()
        self.logger.log_image(key=f'ValImage',
                              images=[fig0,fig1,fig2,fig3],
                              caption=[f'GT T1C Epoch {self.current_epoch}',
                                        f'Energy Epoch {self.current_epoch}',
                                        f'Gn Epoch {self.current_epoch}',
                                        f'Hess Epoch {self.current_epoch}'])
        plt.close(fig0)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        return

    # def on_validation_epoch_end(self):
        # if self.current_epoch % 50 == 0 and self.current_epoch > 0:
        #     Path('models').mkdir(exist_ok=True)
        #     torch.save(self.state_dict(), f'models/energy_manifold_{self.current_epoch}.ckpt')
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        k_hat = self(x)           # [B, 1]
        return k_hat


def dsm_energy_loss(model, u, sigma):
    """
    Energy-based denoising score matching (DSM).

    model: maps x (B,6) -> E(x) (B,1) or (B,)
    u:     clean tuples (B,6)
    sigma: float or tensor broadcastable to (B,6)
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=u.device, dtype=u.dtype)

    # Make sigma broadcastable to u
    if sigma.ndim == 0:
        sigma_view = sigma.view(1, 1)                   # (1,1)
    elif sigma.ndim == 1:
        if sigma.shape[0] == u.shape[0]:
            sigma_view = sigma.view(u.shape[0], 1)      # (B,1)
        elif sigma.shape[0] == u.shape[1]:
            sigma_view = sigma.view(1, u.shape[1])      # (1,6)
        else:
            raise ValueError
    elif sigma.ndim == 2:
        sigma_view = sigma            # (B,1) or (B,6) or (1,6)
    else:
        raise ValueError

    eps = torch.randn_like(u) * sigma_view
    x = (u + eps).detach()
    x.requires_grad_(True)

    E = model(x)
    if E.ndim == 2 and E.shape[1] == 1:
        E_sum = E.sum()
    else:
        E_sum = E.view(-1).sum()

    # ∇_x E(x)
    grad_E = torch.autograd.grad(E_sum, x, create_graph=True)[0]

    # Score implied by energy: s(x) = -∇E(x)
    score = -grad_E

    # Target score for Gaussian corruption q(x|u): ∇_x log q(x|u) = -(x-u)/σ^2 = -eps/σ^2
    target = -eps / (sigma_view ** 2)

    err = (score - target).pow(2).sum(dim=1)     # (B,)
    w = (sigma_view.pow(2).mean(dim=1)).sqrt().pow(2)
    loss = (w * err).mean()
    return loss
'''
Adapted from: https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d_time.py
'''

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from utils import *


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d, self).__init__()
        """
        Initializes the 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Args:
            in_channels (int): input channels to the FNO layer
            out_channels (int): output channels of the FNO layer
            modes (int): number of Fourier modes to multiply, at most floor(N/2) + 1    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor):
        """
        Complex multiplication of the Fourier modes.
        [batch, in_channels, x], [in_channel, out_channels, x] -> [batch, out_channels, x]
            Args:
                input (torch.Tensor): input tensor of size [batch, in_channels, x]
                weights (torch.Tensor): weight tensor of size [in_channels, out_channels, x]
            Returns:
                torch.Tensor: output tensor with shape [batch, out_channels, x]
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor):
        """
        Fourier transformation, multiplication of relevant Fourier modes, backtransformation
        Args:
            x (torch.Tensor): input to forward pass os shape [batch, in_channels, x]
        Returns:
            torch.Tensor: output of size [batch, out_channels, x]
        """
        batchsize = x.shape[0]
        # Fourier transformation
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(pl.LightningModule):
    def __init__(self,hparams):
    
        super().__init__()
        
        self.save_hyperparameters()

        # Training parameters
        self.lr = hparams.lr
        self.weight_decay = hparams.weight_decay
        self.factor = hparams.factor
        self.step_size = hparams.step_size
        self.loss = hparams.loss
        # Model parameters
        self.modes = hparams.modes
        self.width = hparams.width
        self.time_history = hparams.time_history
        self.time_future = hparams.time_future
        self.num_layers = hparams.num_layers
        self.teacher_forcing = hparams.teacher_forcing

        if self.loss == 'l1':
            self.criterion = nn.L1Loss()
        elif self.loss == 'l2':
            self.criterion = nn.MSELoss()
        elif self.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
       
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        self.fc0 = nn.Linear(self.time_history + 2, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.time_future)

        fourier_layers = []
        conv_layers = []
        for i in range(self.num_layers):
            fourier_layers.append(SpectralConv1d(self.width, self.width, self.modes))
            conv_layers.append(nn.Conv1d(self.width, self.width, 1))
        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.conv_layers = nn.ModuleList(conv_layers)

    
    def forward(self, u: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor):
        """
        Forward pass of FNO network.
        The input to the forward pass has the shape [batch, time_history, x].
        1. Add dx and dt as channel dimension to the time_history, repeat for every x
        2. Lift the input to the desired channel dimension by self.fc0
        3. 5 (default) FNO layers
        4. Project from the channel space to the output space by self.fc1 and self.fc2.
        The output has the shape [batch, time_future, x].
        Args:
            u (torch.Tensor): input tensor of shape [batch, time_history, x]
            dx (torch.Tensor): spatial distances
            dt (torch.Tensor): temporal distances
        Returns: torch.Tensor: output has the shape [batch, time_future, x]
        """
        #TODO: rewrite training method and forward pass without permutation
        # [b, x, c] = [b, x, t+2]
        nx = u.shape[1]
        x = torch.cat((u, dx[:, None, None].to(u.device).repeat(1, nx, 1),
                       dt[:, None, None].repeat(1, nx, 1).to(u.device)), -1)

        x = self.fc0(x)
        # [b, x, c] -> [b, c, x]
        x = x.permute(0, 2, 1)

        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)

        # [b, c, x] -> [b, x, c]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.factor)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }
    
    def training_step(self, train_batch, batch_idx):
        u, dx, dt = train_batch
        u = u.float()
        dx = dx.float()
        dt = dt.float()

        L = u.shape[-1]
        u_history = u[:,:self.time_history] # B, T_history, L
        u_future = u[:,self.time_history:] # B, T_future, L
        T_future = u_future.shape[1]

        u_hat = []
        inp = u_history
        for t in range(T_future//self.time_future):
            y_hat = self.forward(inp.permute(0,2,1), dx, dt).permute(0,2,1)
            u_hat.append(y_hat)
            if self.teacher_forcing:
                inp = u_future[:,t*self.time_future:(t+1)*self.time_future]
            else:
                inp = y_hat
        
        u_hat = torch.cat(u_hat, dim=1)
        
        loss = self.criterion(u_hat, u_future)
        mae_loss = self.mae_criterion(u_hat, u_future)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae_loss', mae_loss, prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        u, dx, dt = val_batch
        u = u.float()
        dx = dx.float()
        dt = dt.float()
        
        L = u.shape[-1]

        u_history = u[:,:self.time_history] # B, T_history, L
        u_future = u[:,self.time_history:] # B, T_future, L
        T_future = u_future.shape[1]

        u_hat = []
        inp = u_history
        for t in range(T_future//self.time_future):
            y_hat = self.forward(inp.permute(0,2,1), dx, dt).permute(0,2,1)
            u_hat.append(y_hat)
            inp = y_hat
        
        u_hat = torch.cat(u_hat, dim=1)
        
        loss = self.criterion(u_hat, u_future)
        mae_loss = self.mae_criterion(u_hat, u_future)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae_loss', mae_loss, prog_bar=True)
        return loss
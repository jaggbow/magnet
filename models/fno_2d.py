'''
Adapted from: https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d_time.py
'''
import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
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
        self.modes_1 = hparams.modes_1
        self.modes_2 = hparams.modes_2
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

        self.fc0 = nn.Linear(self.time_history + 3, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.time_future)

        fourier_layers = []
        conv_layers = []
        for i in range(self.num_layers):
            fourier_layers.append(SpectralConv2d(self.width, self.width, self.modes_1, self.modes_2))
            conv_layers.append(nn.Conv2d(self.width, self.width, 1))
        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.conv_layers = nn.ModuleList(conv_layers)

    
    def forward(self, u: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, dt: torch.Tensor):
        """
        Forward pass of FNO network.
        The input to the forward pass has the shape [batch, time_history, H, W].
        1. Add dx, dy and dt as channel dimension to the time_history
        2. Lift the input to the desired channel dimension by self.fc0
        3. 5 (default) FNO layers
        4. Project from the channel space to the output space by self.fc1 and self.fc2.
        The output has the shape [batch, time_future, H, W].
        Args:
            u (torch.Tensor): input tensor of shape [batch, time_history, H, W]
            dx (torch.Tensor): spatial distances
            dy (torch.Tensor): spatial distances
            dt (torch.Tensor): temporal distances
        Returns: torch.Tensor: output has the shape [batch, time_future, x]
        """
        B, T, H, W = u.shape
        x = torch.cat((
            u, 
            dx[:, None, None, None].to(u.device).repeat(1, 1, H, W),
            dy[:, None, None, None].to(u.device).repeat(1, 1, H, W),
            dt[:, None, None, None].repeat(1, 1, H, W).to(u.device)), 1)
        
        x = x.permute(0,2,3,1)
        x = self.fc0(x) # B, H, W, C
        x = x.permute(0, 3, 1, 2) # B, C, H, W

        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)

        x = x.permute(0,2,3,1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
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
        u, dx, dy, dt = train_batch
        u = u.float()
        dx = dx.float()
        dt = dt.float()

        u_history = u[:,:self.time_history] # B, T_history, H, W
        u_future = u[:,self.time_history:] # B, T_future, H, W
        T_future = u_future.shape[1]

        u_hat = []
        inp = u_history
        for t in range(T_future//self.time_future):
            y_hat = self.forward(inp, dx, dy, dt)
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
        u, dx, dy, dt = val_batch
        u = u.float()
        dx = dx.float()
        dt = dt.float()
        
        u_history = u[:,:self.time_history] # B, T_history, H, W
        u_future = u[:,self.time_history:] # B, T_future, H, W
        T_future = u_future.shape[1]

        u_hat = []
        inp = u_history
        for t in range(T_future//self.time_future):
            y_hat = self.forward(inp, dx, dy, dt)
            u_hat.append(y_hat)
            inp = y_hat
        
        u_hat = torch.cat(u_hat, dim=1)
        
        loss = self.criterion(u_hat, u_future)
        mae_loss = self.mae_criterion(u_hat, u_future)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae_loss', mae_loss, prog_bar=True)
        return loss
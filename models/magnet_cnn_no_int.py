import numpy as  np

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from models.backbones.edsr import EDSR
from models.backbones.mlp import MLP
from utils import *


class MAgNetCNN_no_interaction(pl.LightningModule):
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
        self.time_slice = hparams.time_slice
        self.use_lstm = hparams.use_lstm
        self.lstm_hidden = hparams.lstm_hidden
        self.lstm_layers = hparams.lstm_layers
        self.mlp_layers = hparams.mlp_layers
        self.mlp_hidden = hparams.mlp_hidden
        self.scales = hparams.scales
        self.teacher_forcing = hparams.teacher_forcing
        self.res_layers = hparams.res_layers
        self.n_chan = hparams.n_chan
        self.kernel_size = hparams.kernel_size
        self.res_scale = hparams.res_scale
        self.interpolation = hparams.interpolation

        if self.loss == 'l1':
            self.criterion = nn.L1Loss()
        elif self.loss == 'l2':
            self.criterion = nn.MSELoss()
        elif self.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
       
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        self.encoder = EDSR(**{
                "res_layers": self.res_layers,
                "n_chan": self.n_chan,
                "kernel_size": self.kernel_size,
                "res_scale": self.res_scale,
                "mode": "1d",
                "in_chan": self.time_slice})

        self.proj_head = nn.Linear(self.encoder.out_dim+3+1+self.lstm_hidden, self.lstm_hidden)
        
        if self.use_lstm:
            self.lstm_encoder = nn.LSTM(2+self.lstm_hidden, self.lstm_hidden, self.lstm_layers, batch_first=True)
            self.lstm_decoder = nn.LSTM(2*self.lstm_hidden, self.lstm_hidden, self.lstm_layers, batch_first=True)
            self.attn = nn.Sequential(
                nn.Linear(3*self.lstm_hidden, self.lstm_hidden),
                nn.Tanh(),
                nn.Linear(self.lstm_hidden, 1, bias=False))

            self.layernorm = nn.LayerNorm(self.lstm_hidden)
        
            self.decoder = MLP(
                            in_dim=self.lstm_hidden, 
                            hidden_list=[self.mlp_hidden]*self.mlp_layers, 
                            out_dim=1
                        )
        else:
            self.decoder = MLP(
                            in_dim=self.lstm_hidden, 
                            hidden_list=[self.mlp_hidden]*self.mlp_layers, 
                            out_dim=1
                        )        

    def att_decoder(self, inp, hidden, encoder_states):
    
        seq_len = encoder_states.shape[1]
        hidden_ = torch.cat([hidden[0][-1:], hidden[1][-1:]], dim=-1).permute(1,0,2)
        hidden_ = hidden_.repeat(1,seq_len,1)
        alignment_scores = self.attn(torch.cat((hidden_, encoder_states), dim = -1)).squeeze(-1) # shape: [batch, time_in, feat_dim]
        alignment_weights = F.softmax(alignment_scores, dim=1).unsqueeze(1) # shape: [batch, 1, time_in]
        context = torch.bmm(alignment_weights, encoder_states) # shape: [batch, 1, feat_dim]

        inp_decoder = torch.cat([inp, context], dim=-1)
        output, hidden = self.lstm_decoder(inp_decoder, hidden)

        return output, hidden

    def seq2seq_attention(self, x, future_step=1, hidden=None):
        '''
        Args:
            x, [batch,time_in,in_dim]: Input sequence
        '''
        # Encode the input sequence
        encoder_states, hidden = self.lstm_encoder(x, hidden) 

        inp = encoder_states[:,-1:] # [batch, 1, feat_dim]
        outputs = []
        for _ in range(future_step):
            output, hidden = self.att_decoder(inp, hidden, encoder_states)
            outputs.append(output)
            inp = output
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden
    
    def pos_encoding(self, coords):
        '''
        Args:
            coords, [batch, N_coords, 2]: 2D coordinates
            enc_dim, int: Encoding dimension
        '''
        x_proj = (2.*np.pi*coords) @ (torch.eye(1).to(coords.device))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def continuous_decoder(self, x_t, feat, cell, coord_hr, t):
        '''
        Args:
            x_t, [B, T, C, L]
            feat, [B, C, L]: Feature maps
            cell, [B, N, 1]
            coord_hr, [B, N, 1]
            t, [B, T_in+T_out]
        '''
        B, C, L = feat.shape
        T = x_t.shape[1]
        N_coords = coord_hr.shape[1]
        
        # Coordinates in the feature map
        feat_coord = make_coord([L], flatten=False).to(feat.device).permute(1,0).unsqueeze(0).expand(B, 1, L)
        feat_coord = torch.cat([feat_coord, torch.zeros_like(feat_coord)], dim=1)
        feat_coord = feat_coord.unsqueeze(2) # B, 2, 1, L
        
        dx = 1/L
        
        tt = t.unsqueeze(1).repeat(1,cell.shape[1],1) # B, N, T
        
        
        output = []
        latent = torch.randn((B, N_coords, self.lstm_hidden)).to(x_t.device)
        for i in range(T):
            pred_signals = []
            areas = []
            for vx in [-1,1]:
                
                coord = coord_hr.clone().unsqueeze(1) 
                coord = torch.cat([coord, torch.zeros_like(coord)], dim=-1) # B, 1, N, 2
                coord[:, :, :, 0] += vx * dx + 1e-6
                coord.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                # latent code z*
                q_feat = F.grid_sample(feat.unsqueeze(2), coord, mode='nearest', padding_mode="border", align_corners=False)[:,:,0].permute(0,2,1) # B, N, C
                # coordinates
                q_coord = F.grid_sample(feat_coord, coord, mode='nearest', padding_mode="border", align_corners=False)[:,0].permute(0,2,1) # B, N, 1
                # final coordinates
                final_coord = coord_hr-q_coord
                final_coord *= L
                # cell decoding
                final_cell = cell.clone()
                final_cell *= L
                
                areas.append(torch.abs(final_coord).reshape(-1,1)) # B*N, 1
                    
                # true solution
                q_inp = F.grid_sample(x_t[:,i].unsqueeze(2), coord, mode='nearest', padding_mode="border", align_corners=False)[:,:,0].permute(0,2,1) # B, N, C
                # putting all inputs together
                final_input = torch.cat([q_feat, q_inp, final_coord, final_cell, latent, tt[:,:,i:i+1]], dim=-1)
                final_input = final_input.view(B*N_coords, -1)
                
                latent = self.proj_head(final_input) # B*N, C
                pred_signals.append(latent)
                latent = latent.reshape(B, N_coords, -1)
            
            # Area Interpolation
            if self.interpolation == 'area':
                ret = (pred_signals[0]*areas[1]+pred_signals[1]*areas[0])/(areas[1]+areas[0])
            else:
                ret = (pred_signals[0]*areas[1]+pred_signals[1]*areas[0])/(areas[1]+areas[0])
            output.append(ret)
        output = torch.stack(output, dim=1)
        
        return output
    
    def feature_encoding(self, x_t, scale=1):
        B, T, C, L = x_t.shape
        # Encoding x_lr and getting feature maps
        x_t = x_t.reshape(B, T*C, L)
        x_lr = F.interpolate(x_t, size=(L // 2**scale), mode='linear', align_corners=False)
   
        feat = self.encoder(x_lr)

        x = x_lr.reshape(B, T, C, -1)
        return feat, x
    
    def forward(
        self, 
        x_t, 
        coords, 
        cell, 
        t, 
        hr_last,
        hiddens=None):
        '''
        Args:
            x_lr: tensor of shape [B, T, C, L] that represents the low-resolution frames
            coord_hr: tensor of shape [B, N, 1] that represents the N coordinates for sequence in the batch
            t: tensor of shape [B, T] represents the time-coordinates for each sequence in the batch
            hiddens: list of four hidden states to be fed to the LSTM
        '''
        B, T = x_t.shape[:2]
        N_coords = coords.shape[1]
        T_out = t.shape[-1] - T

        z = 0
        for s in range(1,self.scales+1):
            feat, x_lr = self.feature_encoding(x_t, scale=s)
            z +=self.continuous_decoder(x_lr, feat, cell, coords, t)
        z = torch.cat([z, self.pos_encoding(coords).reshape(B*N_coords, -1).unsqueeze(1).repeat(1,T,1)], dim=-1)
    
        if self.use_lstm:
            out, hc = self.seq2seq_attention(z, future_step=T_out)
            ret = self.layernorm(out)
        ret = self.decoder(ret)
       
        outputs = []
        tt = t.unsqueeze(1).repeat(1,cell.shape[1],1)

        for i in range(T_out):
            delta_t = tt[:,:,T+i:T+i+1]-tt[:,:,T-1:T]
            op = ret[:,i].view(B, N_coords, -1) 
            outputs.append(hr_last+delta_t*op)
        outputs = torch.stack(outputs, dim=1)
        return outputs, None
    
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
        t = train_batch['t'].float()
        u = train_batch['hr_frames'].float()
        u_values = train_batch['hr_points'].float()
        coords = train_batch['coords'].float()
        cells = train_batch['cells'].float()
        sample_idx = train_batch['sample_idx']

        u_values_future = u_values[:,self.time_slice:] # B, T_future, N, 1
        B, T_future = u_values_future.shape[:2]

        u_values_hat = []
        inp = u[:,:self.time_slice]
        hr_last = u_values[:,self.time_slice-1]

        for i in range(T_future//self.time_slice):
            y_hat, _ = self.forward(inp, coords, cells, t[:,i*self.time_slice:(i+2)*self.time_slice], hr_last)
            u_values_hat.append(y_hat)
            
            inp = u[:,(i+1)*self.time_slice:(i+2)*self.time_slice]

            if self.teacher_forcing:
                hr_last = u_values[:,(i+2)*self.time_slice-1]
            else:
                reshape_y_hat = y_hat.permute(0,1,3,2)

                for b in range(B):  
                    inp[b,:,:,sample_idx[b]] = reshape_y_hat[b]
                hr_last = y_hat[:,-1]
        
        u_values_hat = torch.cat(u_values_hat, dim=1)

        loss = self.criterion(u_values_hat, u_values_future)
        mae_loss = self.mae_criterion(u_values_hat, u_values_future)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae_loss', mae_loss, prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        t = val_batch['t'].float()
        u = val_batch['hr_frames'].float() # B, T, 1, L
        u_values = val_batch['hr_points'].float()
        coords = val_batch['coords'].float()
        cells = val_batch['cells'].float()

        u_values_future = u_values[:,self.time_slice:] # B, T_future, N, 1
        T_future = u_values_future.shape[1]

        u_values_hat = []
        inp = u[:,:self.time_slice]
        hr_last = u_values[:,self.time_slice-1]

        for i in range(T_future//self.time_slice):
            y_hat, _ = self.forward(inp, coords, cells, t[:,i*self.time_slice:(i+2)*self.time_slice], hr_last)
            u_values_hat.append(y_hat)
            
            inp = y_hat.permute(0,1,3,2)
            hr_last = y_hat[:,-1]
        
        u_values_hat = torch.cat(u_values_hat, dim=1)
        loss = self.criterion(u_values_hat, u_values_future)
        mae_loss = self.mae_criterion(u_values_hat, u_values_future)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae_loss', mae_loss, prog_bar=True)
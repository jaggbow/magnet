import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.nn import MessagePassing, radius_graph

from models.backbones.edsr import EDSR
from models.backbones.mlp import MLP
from utils import *

class Encoder(nn.Module):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        mlp_layers,
        mlp_hidden,
    ):
        super(Encoder, self).__init__()
        self.node_fn = nn.Sequential(
            MLP(
                in_dim=node_in, 
                hidden_list=[mlp_hidden]*mlp_layers, 
                out_dim=node_out),
            nn.LayerNorm(node_out)
        )
        self.edge_fn = nn.Sequential(
            MLP(
                in_dim=edge_in, 
                hidden_list=[mlp_hidden]*mlp_layers, 
                out_dim=edge_out
            ),
            nn.LayerNorm(edge_out)
        )

    def forward(self, x, edge_index, e_features): # global_features
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        return self.node_fn(x), self.edge_fn(e_features)

class InteractionNetwork(MessagePassing):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        mlp_layers,
        mlp_hidden,
    ):
        super(InteractionNetwork, self).__init__(aggr='mean')
        self.node_fn = nn.Sequential(
            MLP(
                in_dim=node_in+edge_out, 
                hidden_list=[mlp_hidden]*mlp_layers, 
                out_dim=node_out),
            nn.LayerNorm(node_out)
        )
        self.edge_fn = nn.Sequential(
            MLP(
                in_dim=node_in+node_in+edge_in, 
                hidden_list=[mlp_hidden]*mlp_layers, 
                out_dim=edge_out
            ),
            nn.LayerNorm(edge_out)
        )

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        x_residual = x
        e_features_residual = e_features
        x, e_features = self.propagate(edge_index=edge_index, x=x, e_features=e_features)
        return x+x_residual, e_features+e_features_residual

    def message(self, edge_index, x_i, x_j, e_features):

        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_fn(e_features)
        return e_features

    def update(self, x_updated, x, e_features):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, e_features

class Processor(MessagePassing):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Processor, self).__init__(aggr='max')
        self.gnn_stacks = nn.ModuleList([
            InteractionNetwork(
                node_in=node_in, 
                node_out=node_out,
                edge_in=edge_in, 
                edge_out=edge_out,
                mlp_layers=mlp_num_layers,
                mlp_hidden=mlp_hidden_dim,
            ) for _ in range(num_message_passing_steps)])

    def forward(self, x, edge_index, e_features):
        for gnn in self.gnn_stacks:
            x, e_features = gnn(x, edge_index, e_features)
        return x, e_features

class Decoder(nn.Module):
    def __init__(
        self, 
        node_in, 
        node_out,
        mlp_layers,
        mlp_hidden,
    ):
        super(Decoder, self).__init__()

        self.node_fn = MLP(
                in_dim=node_in, 
                hidden_list=[mlp_hidden]*mlp_layers, 
                out_dim=node_out)
        

    def forward(self, x):
        # x: (E, node_in)
        return self.node_fn(x)

class MAgNetCNN_2d(pl.LightningModule):
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
        self.num_message_passing_steps = hparams.num_message_passing_steps
        self.latent_dim = hparams.latent_dim
        self.mlp_layers = hparams.mlp_layers
        self.mlp_hidden = hparams.mlp_hidden
        self.scales = hparams.scales
        self.res_layers = hparams.res_layers
        self.n_chan = hparams.n_chan
        self.kernel_size = hparams.kernel_size
        self.res_scale = hparams.res_scale
        self.interpolation = hparams.interpolation
        self.radius = hparams.radius
        self.teacher_forcing = hparams.teacher_forcing

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
                "mode": "2d",
                "in_chan": self.time_slice})

        self.proj_head = nn.Sequential(
                           MLP(
                            in_dim=self.encoder.out_dim+5+1, 
                            hidden_list=[self.mlp_hidden]*self.mlp_layers, 
                            out_dim=self.n_chan),
                           nn.LayerNorm(self.n_chan)
                       )
        self.projector = MLP(
                in_dim=self.n_chan, 
                hidden_list=[self.mlp_hidden]*self.mlp_layers, 
                out_dim=1)
        
        
        self._encoder = Encoder(
            node_in=self.time_slice+3, 
            node_out=self.latent_dim,
            edge_in=self.time_slice+2, 
            edge_out=self.latent_dim,
            mlp_layers=self.mlp_layers,
            mlp_hidden=self.mlp_hidden,
        )
        self._processor = Processor(
            node_in=self.latent_dim, 
            node_out=self.latent_dim,
            edge_in=self.latent_dim, 
            edge_out=self.latent_dim,
            num_message_passing_steps=self.num_message_passing_steps,
            mlp_num_layers=self.mlp_layers,
            mlp_hidden_dim=self.mlp_hidden,
        )
        self._decoder = Decoder(
            node_in=self.latent_dim,
            node_out=self.time_slice,
            mlp_layers=self.mlp_layers,
            mlp_hidden=self.mlp_hidden,
        )
    
    def continuous_decoder(self, x_t, feat, cell, coord_hr, t):
        '''
        Args:
            x_t, [B, T, C, W, W]
            feat, [B, C, W, W]: Feature maps
            cell, [B, N, 1]
            coord_hr, [B, N, 1]
            t, [B, T_in+T_out]
        '''
        B, C, W, W = feat.shape
        T = x_t.shape[1]
        N_coords = coord_hr.shape[1]
        
        # Coordinates in the feature map
        feat_coord = make_coord([W, W], flatten=False).to(feat.device).permute(2,0,1).unsqueeze(0).expand(B, 2, W, W) # B, 2, H, W
        
        dy = 1/W
        dx = 1/W

        tt = t.unsqueeze(1).repeat(1,cell.shape[1],1) # B, N, T
        pred_signals = []
        areas = []

        for vx in [-1,1]:
            for vy in [-1,1]:
                seq_input = []
                coord = coord_hr.clone()
                coord[:, :, 0] += vx * dx + 1e-6
                coord[:, :, 1] += vy * dy + 1e-6
                coord.clamp_(-1 + 1e-6, 1 - 1e-6)

                # latent code z*
                q_feat = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode='nearest', padding_mode="border", align_corners=False)[:,:,0].permute(0,2,1) # B, N, C
                # coordinates
                q_coord = F.grid_sample(feat_coord, coord.flip(-1).unsqueeze(1), mode='nearest', padding_mode="border", align_corners=False)[:,:,0].permute(0,2,1) # B, N, 1
                # final coordinates
                final_coord = coord_hr-q_coord
                final_coord[:, :, 0] *= W
                final_coord[:, :, 1] *= W
                # cell decoding
                final_cell = cell.clone()
                final_cell[:, :, 0] *= W
                final_cell[:, :, 1] *= W

                sides = final_coord.reshape(B*N_coords, -1)
                sides = sides.unsqueeze(1).repeat(1,T,1)
                area = torch.abs(sides[:, :, 0] * sides[:, :, 1])
                areas.append(area + 1e-9)

                for i in range(T):
                    # true solution
                    q_inp = F.grid_sample(x_t[:,i], coord.flip(-1).unsqueeze(1), mode='nearest', padding_mode="border", align_corners=False)[:,:,0].permute(0,2,1) # B, N, C
                    # putting all inputs together (z, [x,c], t)
                    final_input = torch.cat([q_feat, q_inp, final_coord, final_cell, tt[:,:,i:i+1]], dim=-1)
                    final_input = final_input.view(B*N_coords, -1)
                    seq_input.append(final_input)

                seq_input = torch.stack(seq_input, dim=1) # B*N, T, C
                pred_signals.append(self.proj_head(seq_input))
        
        tot_area = torch.stack(areas).sum(dim=0)
        xxx = areas[0]; areas[0] = areas[3]; areas[3] = xxx
        xxx = areas[1]; areas[1] = areas[2]; areas[2] = xxx
        ret = 0
        for pred, area in zip(pred_signals, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
            
        return ret
    
    def feature_encoding(self, x_t):
        B, T, C, W, W = x_t.shape
        # Encoding x_lr and getting feature maps
        x_t = x_t.reshape(B, T*C, W, W)

        feat = self.encoder(x_t)

        return feat
    
    def _build_graph(self, u, x, t):
        B, N, _ = u.shape

        u_ = u.reshape(B*N, -1)
        x_ = x.reshape(B*N, -1)

        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(B*[N])]).to(self.device)
        edges = radius_graph(x_, batch=batch_ids, r=self.radius, loop=True) # (2, n_edges)
        receivers = edges[0, :]
        senders = edges[1, :]
        edge_index = torch.stack([senders, receivers])
        
        node_features = []
        node_features.append(u_)
        node_features.append(x_)
        node_features.append(t[:,-1:].repeat(N, 1))
        node_features = torch.cat(node_features, dim=-1)
        
        edge_features = []

        edge_features.append((u_[senders]-u_[receivers]))
        edge_features.append((x_[senders]-x_[receivers]))
        edge_features = torch.cat(edge_features, dim=-1)

        return node_features, edge_index, edge_features

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
            x_lr: tensor of shape [B, T, C, W, W] that represents the low-resolution frames
            coord_hr: tensor of shape [B, N, 2] that represents the N coordinates for sequence in the batch
            t: tensor of shape [B, T] represents the time-coordinates for each sequence in the batch
        '''
        B, T, _, W, W = x_t.shape
        N_coords = coords.shape[1]
        T_out = t.shape[-1] - T

        feat = self.feature_encoding(x_t)
        z = self.continuous_decoder(x_t, feat, cell, coords, t) # B*N, T, C
        hr_points = self.projector(z) # B*N, T, 1
        
        # Build Graph
        hr_points = hr_points.reshape(B, N_coords, T, -1) # B, N, T, C
        hr_points = hr_points.reshape(B, N_coords, -1) # B, N, TC
        lr_points = x_t.permute(0,3,4,1,2) # B, W, W, T, C
        lr_points = lr_points.reshape(B, W, W, -1) # B, W, W, TC
        lr_points = lr_points.reshape(B, -1, lr_points.shape[-1]) # B, WW, TC

        lr_coords = make_coord([W, W]).to(feat.device).unsqueeze(0).repeat(B, 1, 1) # B, W*W, 2
        all_coords = torch.cat([lr_coords, coords], dim=1) # B, (W*W+N), 2

        all_feats = torch.cat([lr_points, hr_points], dim=1) # B, (WW+N), TC
        
        node_features, edge_index, edge_features = self._build_graph(all_feats, all_coords, t[:,:T])

        node_features, edge_features = self._encoder(node_features, edge_index, edge_features)
        
        node_features, _ = self._processor(node_features, edge_index, edge_features)
        node_features = self._decoder(node_features) # B*(WW+N), T_out
        ret = node_features.reshape(B, -1, node_features.shape[-1]) # B, (WW+N), T_out

        outputs = []
        tt = t.unsqueeze(1).repeat(1,W*W+cell.shape[1],1)
        
        last_values = torch.cat([x_t[:,-1].permute(0,2,3,1).reshape(B, W*W, -1), hr_last], dim=1) # B, (WW+N), 1

        for i in range(T_out):
            delta_t = tt[:,:,T+i:T+i+1]-tt[:,:,T-1:T]
            op = ret[...,i].unsqueeze(-1) # B, (WW+N), 1
            outputs.append(last_values+delta_t*op)
        
        outputs = torch.stack(outputs, dim=1) # B, T, (WW+N), 1

        out_lr = outputs[:,:,:(W*W)] # B, T, WW, C
        out_lr = out_lr.permute(0,1,3,2) # B, T, C, WW
        out_lr = out_lr.reshape(B, T_out, out_lr.shape[2], W, W) # B, T, C, W, W
        out_hr = outputs[:,:,(W*W):] # B, T, N, C
        hr_points = hr_points.reshape(B, N_coords, T, -1) # B, N, T, C
        hr_points = hr_points.permute(0,2,1,3) # B, N, T, C

        return out_hr, out_lr, hr_points
    
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
        u = train_batch['lr_frames'].float()
        B, T, C, W, W = u.shape
        u_values = train_batch['hr_points'].float()
        coords = train_batch['coords'].float()
        cells = train_batch['cells'].float()

        u_values_future = u_values[:,self.time_slice:] # B, T_future, N, 1
        B, T_future = u_values_future.shape[:2]

        u_values_hat = []
        hr_values_hat = []

        inp = u[:,:self.time_slice]
        hr_last = u_values[:,self.time_slice-1]

        for i in range(T_future//self.time_slice):
            out_hr, out_lr, hr_points = self.forward(inp, coords, cells, t[:,i*self.time_slice:(i+2)*self.time_slice], hr_last)
            y_hat = torch.cat([out_hr, out_lr.reshape(*out_lr.shape[:3], -1).permute(0,1,3,2)], dim=2)
            u_values_hat.append(y_hat)
            hr_values_hat.append(hr_points)
        
            if self.teacher_forcing:
                inp = u[:,(i+1)*self.time_slice:(i+2)*self.time_slice] # B, T, C, W, W
                hr_last = u_values[:,(i+2)*self.time_slice-1]
            else:
                inp = out_lr
                hr_last = out_hr[:,-1]
        
        u_values_hat = torch.cat(u_values_hat, dim=1) # B, T_out, (N+L), 1 
        hr_values_hat = torch.cat(hr_values_hat, dim=1) # B, T_in, N, 1
        

        target = torch.cat([u_values_future, u[:,self.time_slice:].reshape(B, T-self.time_slice, C, -1).permute(0,1,3,2)], dim=2)
        loss = self.criterion(u_values_hat, target)+self.criterion(hr_values_hat, u_values[:,:-self.time_slice])
        mae_loss = self.mae_criterion(u_values_hat, target)
        interp_loss = self.mae_criterion(hr_values_hat, u_values[:,:-self.time_slice])

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae_loss', mae_loss, prog_bar=True)
        self.log('train_interp_loss', interp_loss, prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        t = val_batch['t'].float()
        u = val_batch['lr_frames'].float() # B, T, 1, W, W
        B, T, _, W, W = u.shape
        u_values = val_batch['hr_points'].float()
        coords = val_batch['coords'].float()
        cells = val_batch['cells'].float()

        u_values_future = u_values[:,self.time_slice:] # B, T_future, N, 1
        T_future = u_values_future.shape[1]

        u_values_hat = []
        inp = u[:,:self.time_slice]
        hr_last = u_values[:,self.time_slice-1]

        for i in range(T_future//self.time_slice):
            y_hat, _, _ = self.forward(inp, coords, cells, t[:,i*self.time_slice:(i+2)*self.time_slice], hr_last)            
            u_values_hat.append(y_hat)
            
            inp = y_hat.permute(0,1,3,2) # B, T, C, WW
            W_inp = int(inp.shape[-1]**0.5)
            B, T = inp.shape[:2]
            inp = inp.reshape(-1, *inp.shape[2:]) # BT, C, WW
            inp = inp.reshape(-1, inp.shape[1], W_inp, W_inp) # BT, C, W, W
            inp = F.interpolate(inp, size=W, mode='bilinear', align_corners=False).reshape(B, T, -1, W, W)
            hr_last = y_hat[:,-1]
        
        u_values_hat = torch.cat(u_values_hat, dim=1)
        loss = self.criterion(u_values_hat, u_values_future)
        mae_loss = self.mae_criterion(u_values_hat, u_values_future)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae_loss', mae_loss, prog_bar=True)
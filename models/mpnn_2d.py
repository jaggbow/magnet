import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.nn import MessagePassing, radius_graph, InstanceNorm
from torch_geometric.data import Data

from models.backbones.mlp import MLP
from utils import *

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 n_variables: int):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + 2 + n_variables, hidden_features),
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class MPNN_2d(pl.LightningModule):
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
        self.out_features = hparams.time_window
        self.hidden_features = hparams.hidden_features
        self.hidden_layer = hparams.hidden_layer
        self.time_window = hparams.time_window
        self.teacher_forcing = hparams.teacher_forcing
        self.n = hparams.neighbors

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=1
                                        )
                               )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window + 3, self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==10):
            self.output_mlp = nn.Sequential(
                                            nn.Conv1d(1, 8, 16, stride=6),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1))
        if(self.time_window==16):
            self.output_mlp = nn.Sequential(
                                            nn.Conv1d(1, 8, 16, stride=5),
                                            Swish(),
                                            nn.Conv1d(8, 1, 8, stride=1))
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )

        if self.loss == 'l1':
            self.criterion = nn.L1Loss()
        elif self.loss == 'l2':
            self.criterion = nn.MSELoss()
        elif self.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
       
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
    
    def forward(self, data, L, tmax, dt):

        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / L
        pos_t = pos[:, 0][:, None] / tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        
        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos_x, variables), -1)
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window).to(dt.device) * dt).to(dt.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.factor)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }
    
    def _build_graph(self,
                    data: torch.Tensor,
                    t: torch.Tensor,
                    x: torch.Tensor,
                    steps: list):
        """
        data, [B, T, N]
        t, [B]
        x, [B, N]
        steps, [B]
        """
        nx = data.shape[-1]

        u = torch.Tensor().to(data.device)
        x_pos = torch.Tensor().to(data.device)
        t_pos = torch.Tensor().to(data.device) 
        batch = torch.Tensor().to(data.device)

        for b, (data_batch, step) in enumerate(zip(data, steps)):
            u = torch.cat((u, torch.transpose(torch.cat([d[None, :] for d in data_batch]), 0, 1)), )
            x_pos = torch.cat((x_pos, x[0]), )
            t_pos = torch.cat((t_pos, torch.ones(nx, device=t.device) * t[b, step]), )
            batch = torch.cat((batch, torch.ones(nx, device=batch.device) * b), )

        # Calculate the edge_index
        dx = x[0][1] - x[0][0]
        dy = x[0][int(nx**0.5)] - x[0][0]
        dr = torch.norm(dx-dy, p=2)
        radius = self.n * dr + 0.0001

        edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
        
        graph = Data(x=u, edge_index=edge_index)
        graph.pos = torch.cat((t_pos[:, None], x_pos), 1)
        graph.batch = batch.long()

        return graph

    
    def training_step(self, train_batch, batch_idx):
        u = train_batch['u'].float().permute(0,2,1)
        x = train_batch['x'].float().squeeze(-1)
        B, _, N = u.shape
        t = train_batch['t'].float() # B, T
        dt = t[0][1] - t[0][0]
                
        graph = self._build_graph(
            u[:,:self.time_window,:], 
            t,
            x,
            steps=[self.time_window-1]*B)
        
        target = u[:,self.time_window:,:]
        T_out = target.shape[1]
        
        u_hat = []
        for i in range(T_out//self.time_window):
            y_hat = self.forward(graph, x[0,-1], t[0,-1], dt)
            y_hat = y_hat.reshape(B, N, -1).permute(0,2,1)
            u_hat.append(y_hat)
            
            if self.teacher_forcing:
                graph = self._build_graph(
                        u[:,(i+1)*self.time_window:(i+2)*self.time_window,:], 
                        t,
                        x,
                        steps=[(i+2)*self.time_window-1]*B)
            else:
                graph = self._build_graph(
                        y_hat, 
                        t,
                        x,
                        steps=[(i+2)*self.time_window-1]*B)

        u_hat = torch.cat(u_hat, dim=1)

        loss = self.criterion(u_hat, target)
        mae_loss = self.mae_criterion(u_hat, target)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae_loss', mae_loss, prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        u = val_batch['u'].float().permute(0,2,1)
        x = val_batch['x'].float().squeeze(-1)
        B, T_in, N = u.shape
        t = val_batch['t'].float() # B, T
        dt = t[0][1] - t[0][0]
                
        graph = self._build_graph(
            u[:,:self.time_window,:], 
            t,
            x,
            steps=[self.time_window-1]*B)
        
        target = u[:,self.time_window:,:]
        T_out = target.shape[1]
        
        u_hat = []
        for i in range(T_out//self.time_window):
            y_hat = self.forward(graph, x[0,-1], t[0,-1], dt)
            y_hat = y_hat.reshape(B, N, -1).permute(0,2,1)
            u_hat.append(y_hat)
                        
            graph = self._build_graph(
                        y_hat, 
                        t,
                        x,
                        steps=[(i+2)*self.time_window-1]*B)

        u_hat = torch.cat(u_hat, dim=1)

        loss = self.criterion(u_hat, target)
        mae_loss = self.mae_criterion(u_hat, target)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae_loss', mae_loss, prog_bar=True)

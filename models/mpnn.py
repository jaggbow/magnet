import torch
from torch import nn

import pytorch_lightning as pl

from torch_geometric.nn import MessagePassing, radius_graph

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
        super(InteractionNetwork, self).__init__(aggr='add')
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

class MPNN(pl.LightningModule):
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
        self.node_in = hparams.node_in
        self.latent_dim = hparams.latent_dim
        self.edge_in = hparams.edge_in
        self.mlp_num_layers = hparams.mlp_num_layers
        self.mlp_hidden_dim = hparams.mlp_hidden_dim
        self.num_message_passing_steps = hparams.num_message_passing_steps
        self.node_out = hparams.node_out
        self.radius = hparams.radius
        self.teacher_forcing = hparams.teacher_forcing
        self.time_slice = hparams.time_slice

        if self.loss == 'l1':
            self.criterion = nn.L1Loss()
        elif self.loss == 'l2':
            self.criterion = nn.MSELoss()
        elif self.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
       
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        self._encoder = Encoder(
            node_in=self.node_in, 
            node_out=self.latent_dim,
            edge_in=self.edge_in, 
            edge_out=self.latent_dim,
            mlp_layers=self.mlp_num_layers,
            mlp_hidden=self.mlp_hidden_dim,
        )
        self._processor = Processor(
            node_in=self.latent_dim, 
            node_out=self.latent_dim,
            edge_in=self.latent_dim, 
            edge_out=self.latent_dim,
            num_message_passing_steps=self.num_message_passing_steps,
            mlp_num_layers=self.mlp_num_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )
        self._decoder = Decoder(
            node_in=self.latent_dim,
            node_out=self.node_out,
            mlp_layers=self.mlp_num_layers,
            mlp_hidden=self.mlp_hidden_dim,
        )

    
    def forward(self, x, edge_index, e_features):

        x, e_features = self._encoder(x, edge_index, e_features)
        x, e_features = self._processor(x, edge_index, e_features)
        x = self._decoder(x)
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
    
    def _build_graph(self, u, x, t):
        B, N, T = u.shape

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
    
    def training_step(self, train_batch, batch_idx):
        u = train_batch['u'].float()
        x = train_batch['x'].float() 
        B, N, _ = x.shape
        t = train_batch['t'].float()
                
        node_features, edge_index, edge_features = self._build_graph(u[:,:,:self.time_slice], x, t[:,:self.time_slice])
        target = u[:,:,self.time_slice:]
        T_out = target.shape[-1]
        u_hat = []
        for i in range(T_out//self.time_slice):
            y_hat = self.forward(node_features, edge_index, edge_features)
            u_hat.append(y_hat)
            if self.teacher_forcing:
                inp = target[:,:,i*self.time_slice:(i+1)*self.time_slice]
                inp = inp.reshape(B*N, -1)
            else:
                inp = y_hat
            node_features = torch.cat([inp, x.reshape(B*N, -1), t[:, (i+2)*self.time_slice-1:(i+2)*self.time_slice].repeat(N, 1)], dim=-1)

        
        u_hat = torch.cat(u_hat, dim=-1)
        u_hat = u_hat.reshape(B, N, -1)
        
        loss = self.criterion(u_hat, target)
        mae_loss = self.mae_criterion(u_hat, target)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae_loss', mae_loss, prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        u = val_batch['u'].float()
        x = val_batch['x'].float() 
        B, N, _ = x.shape
        t = val_batch['t'].float()
                
        node_features, edge_index, edge_features = self._build_graph(u[:,:,:self.time_slice], x, t[:,:self.time_slice])
        target = u[:,:,self.time_slice:]
        T_out = target.shape[-1]
        u_hat = []
        for i in range(T_out//self.time_slice):
            y_hat = self.forward(node_features, edge_index, edge_features)
            u_hat.append(y_hat)
            node_features = torch.cat([y_hat, x.reshape(B*N, -1), t[:, (i+2)*self.time_slice-1:(i+2)*self.time_slice].repeat(N, 1)], dim=-1)
            # inp = u_future[:,t*self.time_slice:(t+1)*self.time_slice]
            # inp = y_hat
        
        u_hat = torch.cat(u_hat, dim=-1)
        u_hat = u_hat.reshape(B, N, -1)
        
        loss = self.criterion(u_hat, target)
        mae_loss = self.mae_criterion(u_hat, target)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae_loss', mae_loss, prog_bar=True)
    
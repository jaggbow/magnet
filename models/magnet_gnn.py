import torch
from torch import nn

import pytorch_lightning as pl

from torch_geometric.nn import MessagePassing, radius_graph, knn

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
                out_dim=edge_out,
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
                nn.LayerNorm(node_out))
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

class MAgNetGNN(pl.LightningModule):
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
        self.n_chan = hparams.n_chan
        self.radius = hparams.radius
        self.codec_neighbors = hparams.codec_neighbors
        self.teacher_forcing = hparams.teacher_forcing
        self.noise = hparams.noise
        self.interpolation = hparams.interpolation

        if self.loss == 'l1':
            self.criterion = nn.L1Loss()
        elif self.loss == 'l2':
            self.criterion = nn.MSELoss()
        elif self.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
       
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        self.encoder = Encoder(
            node_in=self.time_slice+2, 
            node_out=self.latent_dim,
            edge_in=self.time_slice+1, 
            edge_out=self.latent_dim,
            mlp_layers=self.mlp_layers,
            mlp_hidden=self.mlp_hidden,
        )
        self.processor = Processor(
            node_in=self.latent_dim, 
            node_out=self.latent_dim,
            edge_in=self.latent_dim, 
            edge_out=self.latent_dim,
            num_message_passing_steps=self.num_message_passing_steps,
            mlp_num_layers=self.mlp_layers,
            mlp_hidden_dim=self.mlp_hidden,
        )

        self.proj_head = nn.Linear(self.latent_dim+3, self.n_chan)
        self.projector = MLP(
                in_dim=self.n_chan, 
                hidden_list=[self.mlp_hidden]*self.mlp_layers, 
                out_dim=1)
        
        
        self._encoder = Encoder(
            node_in=self.time_slice+2, 
            node_out=self.latent_dim,
            edge_in=self.time_slice+1, 
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
    
    def continuous_decoder(
        self,
        x_lr, 
        lr_encoded, 
        lr_coords, 
        hr_coords, 
        t):
        '''
        Args:
            x_lr, [B, T, C, L]
            lr_encoded, [B, L, C]: 
            lr_coords, [B, L, 1]
            hr_coords, [B, N, 1]
            t, [B, T]
        '''
        B, T, _, L = x_lr.shape
        N = hr_coords.shape[1]

        # Find nearest k low-res neighbors for each high-res coordinate (k=2 by default)
        flat_lr_coords = lr_coords.reshape(B*L, -1)
        batch_lr = torch.cat([torch.LongTensor([i]*L) for i in range(B)]).to(flat_lr_coords.device)
        flat_hr_coords = hr_coords.reshape(B*N, -1)
        batch_hr = torch.cat([torch.LongTensor([i]*N) for i in range(B)]).to(flat_hr_coords.device)
        assign_index = knn(flat_lr_coords, flat_hr_coords, self.codec_neighbors, batch_lr, batch_hr)

        lr_encoded_flat = lr_encoded.reshape(B*L, -1)
        timesteps = t.unsqueeze(1).repeat(1,N,1) # B, N, T
        timesteps = timesteps.reshape(B*N, -1) # B*N, T

        out = []
        for i in range(T):
            weights = []
            latents = []
            x_lr_flat = x_lr[:,i].permute(0,2,1).reshape(B*L, -1)
            timestep = timesteps[:,i:i+1]
            for j in range(self.codec_neighbors):
                q_feat = lr_encoded_flat[assign_index[1,j::self.codec_neighbors]]
                q_inp = x_lr_flat[assign_index[1,j::self.codec_neighbors]]
                q_coord = flat_lr_coords[assign_index[1,j::self.codec_neighbors]]
                final_coord = q_coord-flat_hr_coords

                final_input = torch.cat([q_feat, q_inp, final_coord, timestep], dim=-1)
                if self.interpolation == 'area':
                    weight = torch.abs(final_coord) # B*N, 1
                elif self.interpolation == 'knn':
                    weight = (1/(torch.norm(final_coord, 2, dim=-1)**2)).unsqueeze(-1)
                elif self.interpolation == 'sph':
                    weight = torch.pow(1 - (L*torch.norm(final_coord, 2, dim=-1)**2), 3).unsqueeze(-1)
                latents.append(self.proj_head(final_input)) # B*N, C
                weights.append(weight)
            
            if self.interpolation == 'area':
                latent = (latents[0]*weights[1]+latents[1]*weights[0])/(weights[1]+weights[0])
            else:
                latent = (latents[0]*weights[0]+latents[1]*weights[1])/(weights[1]+weights[0])
            out.append(latent)
        
        out = torch.stack(out, dim=1) # B*N, T, C
        return out

    
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
        x_lr,
        lr_coords, 
        hr_coords, 
        t, 
        hr_last):
        '''
        Args:
            x_lr: tensor of shape [B, T, C, L] that represents the low-resolution frames
            lr_coords: tensor of shape [B, L, 1] that represents the L coordinates for sequence of low frames in the batch
            hr_coords: tensor of shape [B, N, 1] that represents the N coordinates for sequence of points in the batch
            t: tensor of shape [B, T] represents the time-coordinates for each sequence in the batch
        '''
        B, T, C, L = x_lr.shape
        N = hr_coords.shape[1]
        T_out = t.shape[1] - T

        # Build graph and encode it
        u = x_lr.permute(0,3,1,2) # B, L, T, C
        u = u.reshape(B, L, -1) # B, L, C
        node_features, edge_index, edge_features = self._build_graph(u, lr_coords, t[:,:T])
        node_features, edge_features = self.encoder(node_features, edge_index, edge_features)
        lr_encoded, _ = self.processor(node_features, edge_index, edge_features)

        # Get interpolated features from low-res points
        z = self.continuous_decoder(x_lr, lr_encoded, lr_coords, hr_coords, t) # B*N, T, C
        hr_points = self.projector(z) # B*N, T, 1
        
        # Build Graph
        hr_points = hr_points.reshape(B, N, T, -1) # B, N, T, C
        hr_points = hr_points.reshape(B, N, -1) # B, N, C
        lr_points = x_lr.permute(0,3,1,2) # B, L, T, C
        lr_points = lr_points.reshape(B, L, -1) # B, L, C

        all_coords = torch.cat([lr_coords, hr_coords], dim=1) # B, (L+N), 1

        all_feats = torch.cat([lr_points, hr_points], dim=1) # B, (L+N), C

        node_features, edge_index, edge_features = self._build_graph(all_feats, all_coords, t[:,:T])


        node_features, edge_features = self._encoder(node_features, edge_index, edge_features)
        node_features, _ = self._processor(node_features, edge_index, edge_features)
        node_features = self._decoder(node_features) # B*(L+N), T_out
        ret = node_features.reshape(B, -1, node_features.shape[-1]) # B, (L+N), T_out

        outputs = []
        tt = t.unsqueeze(1).repeat(1,L+N,1)

        last_values = torch.cat([x_lr[:,-1].permute(0,2,1), hr_last], dim=1) # B, (L+N), 1

        for i in range(T_out):
            delta_t = tt[:,:,T+i:T+i+1]-tt[:,:,T-1:T]
            op = ret[...,i].unsqueeze(-1) # B, (L+N), 1
            outputs.append(last_values+delta_t*op)
        
        outputs = torch.stack(outputs, dim=1) # B, T, (L+N), 1

        out_lr = outputs[:,:,:L]
        out_hr = outputs[:,:,L:]
        hr_points = hr_points.reshape(B, N, T, -1)
        hr_points = hr_points.permute(0,2,1,3)

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
        u_values = train_batch['hr_points'].float()
        coords = train_batch['coords_hr'].float()
        lr_coords = train_batch['coords_lr'].float()

        u_values_future = u_values[:,self.time_slice:] # B, T_future, N, 1
        B, T_future = u_values_future.shape[:2]

        u_values_hat = []
        hr_values_hat = []

        inp = u[:,:self.time_slice]
        noise = self.noise*torch.randn(inp.shape).to(inp.device)
        inp = inp+noise

        hr_last = u_values[:,self.time_slice-1]
        noise = self.noise*torch.randn(hr_last.shape).to(hr_last.device)
        hr_last = hr_last+noise

        for i in range(T_future//self.time_slice):
            out_hr, out_lr, hr_points = self.forward(inp, lr_coords, coords, t[:,i*self.time_slice:(i+2)*self.time_slice], hr_last)
            y_hat = torch.cat([out_hr, out_lr], dim=2)
            u_values_hat.append(y_hat)
            hr_values_hat.append(hr_points)
        
            if self.teacher_forcing:
                inp = u[:,(i+1)*self.time_slice:(i+2)*self.time_slice] # B, T, C, L
                hr_last = u_values[:,(i+2)*self.time_slice-1]
            else:
                inp = out_lr.permute(0,1,3,2)
                hr_last = out_hr[:,-1]

            noise = self.noise*torch.randn(inp.shape).to(inp.device)
            inp = inp+noise

            noise = self.noise*torch.randn(hr_last.shape).to(hr_last.device)
            hr_last = hr_last+noise
        
        u_values_hat = torch.cat(u_values_hat, dim=1) # B, T_out, (N+L), 1 
        hr_values_hat = torch.cat(hr_values_hat, dim=1) # B, T_in, N, 1
        
        target = torch.cat([u_values_future, u[:,self.time_slice:].permute(0,1,3,2)], dim=2)
        loss = self.criterion(u_values_hat, target)+self.criterion(hr_values_hat, u_values[:,:-self.time_slice])
        mae_loss = self.mae_criterion(u_values_hat, target)
        interp_loss = self.mae_criterion(hr_values_hat, u_values[:,:-self.time_slice])

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae_loss', mae_loss, prog_bar=True)
        self.log('train_interp_loss', interp_loss, prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        t = val_batch['t'].float()
        u = val_batch['lr_frames'].float() # B, T, 1, L
        u_values = val_batch['hr_points'].float()
        coords = val_batch['coords_hr'].float()
        lr_coords = val_batch['coords_lr'].float()

        u_values_future = u_values[:,self.time_slice:] # B, T_future, N, 1
        T_future = u_values_future.shape[1]

        u_values_hat = []
        inp = u[:,:self.time_slice]
        hr_last = u_values[:,self.time_slice-1]

        for i in range(T_future//self.time_slice):
            out_hr, out_lr, _ = self.forward(
                inp, 
                lr_coords, 
                coords, 
                t[:,i*self.time_slice:(i+2)*self.time_slice], 
                hr_last)
            y_hat = torch.cat([out_hr, out_lr], dim=2)
            u_values_hat.append(y_hat)
            
            inp = out_lr.permute(0,1,3,2)
            hr_last = out_hr[:,-1]
        
        u_values_hat = torch.cat(u_values_hat, dim=1)
        target = torch.cat([u_values_future, u[:,self.time_slice:].permute(0,1,3,2)], dim=2)
        loss = self.criterion(u_values_hat, target)
        mae_loss = self.mae_criterion(u_values_hat, target)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae_loss', mae_loss, prog_bar=True)
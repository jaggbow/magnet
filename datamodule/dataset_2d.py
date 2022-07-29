import numpy as np
import h5py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from torch_geometric.nn import radius_graph

from utils import *

class HDF5DatasetGraph(Dataset):
    
    def __init__(self, 
                 path,
                 nt,
                 nx,
                 mode='train', 
                 load_all=False,
                 in_timesteps=16,
                 radius=2):
        
        assert mode in ['train', 'valid', 'test'], "mode must belong to one of these ['train', 'val', 'test']"
        
        f = h5py.File(path, 'r')
        self.mode = mode
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{nx}'
        self.radius = radius
        self.in_timesteps = in_timesteps

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx):
        
        x = torch.from_numpy(self.data['x'][idx]).unsqueeze(-1) # N, 1
        t = torch.from_numpy(self.data['t'][idx]) # T
        u = torch.from_numpy(self.data[self.dataset][idx]).permute(1,0) # N, T
        
        return_tensors = {
            'u': u,
            'x': x,
            't': t
        }
        return return_tensors

class HDF5DatasetImplicitGNN(Dataset):
    
    def __init__(self, 
                 path,
                 nt,
                 nx,
                 sampling='uniform',
                 mode='train', 
                 load_all=False,
                 samples = 256):
        
        assert mode in ['train', 'valid', 'test'], "mode must belong to one of these ['train', 'val', 'test']"
        
        f = h5py.File(path, 'r')
        self.mode = mode
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{nx}'
        self.samples = samples
        self.sampling = sampling

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx):
        
        x = self.data['x'][idx]
        # Normalize time coordinates
        x = 2*(x-x.min())/(x.max()-x.min())-1
        
        t = self.data['t'][idx]
        u_hr = torch.from_numpy(self.data[self.dataset][idx]).unsqueeze(1) # T, 1, L
        T, _, L = u_hr.shape
        u_lr = u_hr[:,:,::2] # T, 1, L//2
        lr_coord = x[::2]

        if self.mode in ['train']:
            indices_left = np.setdiff1d(np.arange(0,L), np.arange(0,L)[::2])
            sample_lst = torch.tensor(sorted(np.random.choice(indices_left, self.samples, replace=False)))
            hr_coord = x[sample_lst]

            hr_points = u_hr[:,:,sample_lst].permute(0,2,1)

            return_tensors = {
            't': t,
            'sample_idx': sample_lst,
            'lr_frames': u_lr,
            'hr_frames': u_hr,
            'hr_points': hr_points, 
            'coords_hr': hr_coord,
            'coords_lr': lr_coord
            }
        else:
            indices_left = np.setdiff1d(np.arange(0,L), np.arange(0,L)[::2])
            hr_coord = x[indices_left]

            hr_points = u_hr[:,:,indices_left].permute(0,2,1)

            return_tensors = {
            't': t,
            'lr_frames': u_lr,
            'hr_frames': u_hr,
            'hr_points': hr_points, 
            'coords_hr': hr_coord,
            'coords_lr': lr_coord 
        }

        return return_tensors

class HDF5DatasetImplicit(Dataset):
    
    def __init__(self, 
                 path,
                 nt,
                 nx,
                 sampling='uniform',
                 mode='train', 
                 load_all=False,
                 samples = 256):
        
        assert mode in ['train', 'valid', 'test'], "mode must belong to one of these ['train', 'val', 'test']"
        
        f = h5py.File(path, 'r')
        self.mode = mode
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{nx}'
        self.samples = samples
        self.sampling = sampling

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx):
        
        x = self.data['x'][idx]
        t = self.data['t'][idx]
        u_hr = torch.from_numpy(self.data[self.dataset][idx]).unsqueeze(1) # T, 1, L

        T, _, L = u_hr.shape
        u_lr = F.interpolate(u_hr, size=(L // 2), mode='linear', align_corners=False) # T, 1, L//2

        if self.mode in ['train']:
            if self.sampling == 'uniform':
                sample_lst = torch.tensor(sorted(np.random.choice(L, self.samples, replace=False)))
            elif self.sampling == 'boundary':
                p = torch.softmax(torch.pow(torch.abs(torch.arange(L)-L//2)/L, 2)/0.1, dim=0).numpy()
                sample_lst = torch.tensor(sorted(np.random.choice(L, self.samples, p=p, replace=False)))
            hr_coord = make_coord([L])[sample_lst]

            cell = torch.ones_like(hr_coord)
            cell *= 2 / L
            hr_points = torch.stack([to_pixel_samples(u_hr[i])[1][sample_lst] for i in range(T)], dim=0)

            return_tensors = {
            't': t,
            'sample_idx': sample_lst,
            'lr_frames': u_lr,
            'hr_frames': u_hr,
            'hr_points': hr_points, 
            'coords': hr_coord, 
            'cells': cell
        }
        else:
            hr_coord = make_coord([L])

            cell = torch.ones_like(hr_coord)
            cell *= 2 / L
            hr_points = torch.stack([to_pixel_samples(u_hr[i])[1] for i in range(T)], dim=0)

            return_tensors = {
            't': t,
            'lr_frames': u_lr,
            'hr_frames': u_hr,
            'hr_points': hr_points, 
            'coords': hr_coord, 
            'cells': cell
        }

        return return_tensors



class HDF5Dataset(Dataset):
    """
    Load samples of an PDE Dataset, get items according to PDE.
    """
    def __init__(self, 
                 path: str,
                 mode: str,
                 nt: int,
                 res: int,
                 dtype=torch.float32,
                 load_all: bool=False):
        """Initialize the dataset object.
        Args:
            path: path to dataset
            mode: [train, valid, test]
            nt: temporal resolution
            res: spatial resolution
            shift: [fourier, linear]
            dtype: floating precision of data
            load_all: load all the data into memory
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.dtype = dtype
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{res}'

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx: int):
        """
        Returns data items for batched training/validation/testing.
        Args:
            idx: data index
        Returns:
            torch.Tensor: data trajectory used for training/validation/testing
            torch.Tensor: dx
            torch.Tensor: dt
        """
        u = torch.from_numpy(self.data[self.dataset][idx])
        dx = torch.from_numpy(self.data['dx'][idx])[0]
        dy = torch.from_numpy(self.data['dy'][idx])[0]
        dt = torch.from_numpy(self.data['dt'][idx])[0]

        return u, dx, dy, dt
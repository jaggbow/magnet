import numpy as np
import h5py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from torch_geometric.nn import radius_graph

from utils import *

class HDF5DatasetGraph_2d(Dataset):
    
    def __init__(self, 
                 path,
                 nt,
                 res,
                 mode='train', 
                 regular=True,
                 load_all=False):
        
        assert mode in ['train', 'valid', 'test'], "mode must belong to one of these ['train', 'val', 'test']"
        
        f = h5py.File(path, 'r')
        self.mode = mode
        self.data = f[self.mode]
        self.regular = regular
        self.dataset = f'pde_{nt}-{res}'

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx):
        
        u = torch.from_numpy(self.data[self.dataset][idx]) # T, W, W
        W = u.shape[-1]
        u = u.reshape(u.shape[0], -1) # T, WW
        u = u.permute(1,0) # WW, T

        if self.regular:
            x = torch.from_numpy(self.data['x'][idx])
            y = torch.from_numpy(self.data['y'][idx])
            coords = torch.stack(torch.meshgrid(x,y), dim=-1).reshape(-1,2)
        else:
            coords = torch.from_numpy(self.data['coords'][idx])
        t = torch.from_numpy(self.data['t'][idx]) # T
        
        
        return_tensors = {
            'u': u,
            'x': coords,
            't': t
        }
        return return_tensors

class HDF5DatasetImplicitGNN_2d(Dataset):
    
    def __init__(self, 
                 path,
                 nt,
                 res,
                 mode='train',
                 regular=True,
                 load_all=False,
                 samples = 256):
        
        assert mode in ['train', 'valid', 'test'], "mode must belong to one of these ['train', 'val', 'test']"
        
        f = h5py.File(path, 'r')
        self.mode = mode
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{res}'
        self.samples = samples
        self.regular = regular

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx):
        
        if self.regular:
            x = self.data['x'][idx]
            y = self.data['y'][idx]
            coords = np.stack(np.meshgrid(x,y), axis=-1)
            coords = coords.reshape(-1, coords.shape[-1])
            u_hr = torch.from_numpy(self.data[self.dataset][idx]).unsqueeze(1) # T, 1, W, W
            u_hr = u_hr.reshape(u_hr.shape[0], 1, -1)
        else:
            coords = self.data['coords'][idx] # N, 2
            u_hr = torch.from_numpy(self.data[self.dataset][idx]).unsqueeze(1) # T, 1, N
        coords = 2*(coords-coords.min(0))/(coords.max(0)-coords.min(0))-1
        
        t = self.data['t'][idx]
        
        T, _, N = u_hr.shape
        u_lr = u_hr[:,:,::2] # T, 1, N//2
        lr_coord = coords[::2]

        if self.mode in ['train']:
            indices_left = np.setdiff1d(np.arange(0,N), np.arange(0,N)[::2])
            sample_lst = torch.tensor(sorted(np.random.choice(indices_left, self.samples, replace=False)))
            hr_coord = coords[sample_lst]

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
            indices_left = np.setdiff1d(np.arange(0,N), np.arange(0,N)[::2])
            hr_coord = coords[indices_left]

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

class HDF5DatasetImplicit_2d(Dataset):
    
    def __init__(self, 
                 path,
                 nt,
                 res,
                 mode='train', 
                 load_all=False,
                 samples = 256):
        
        assert mode in ['train', 'valid', 'test'], "mode must belong to one of these ['train', 'val', 'test']"
        
        f = h5py.File(path, 'r')
        self.mode = mode
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{res}'
        self.samples = samples

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx):
        
        t = self.data['t'][idx]
        u_hr = torch.from_numpy(self.data[self.dataset][idx]).unsqueeze(1) # T, 1, W, W

        T, _, W, W = u_hr.shape
        u_lr = F.interpolate(u_hr, size=(W // 2), mode='bilinear', align_corners=False) # T, 1, W//2, W//2

        if self.mode in ['train']:
            sample_lst = torch.tensor(sorted(np.random.choice(W*W, self.samples, replace=False)))
    
            hr_coord = make_coord([W, W])[sample_lst]

            cell = torch.ones_like(hr_coord)
            cell *= 2 / W
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
            hr_coord = make_coord([W, W])

            cell = torch.ones_like(hr_coord)
            cell *= 2 / W
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
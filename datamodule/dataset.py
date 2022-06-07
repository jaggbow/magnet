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

class HDF5DatasetCon(Dataset):
    
    def __init__(self, 
                 path,
                 nt,
                 nx,
                 mode='train', 
                 load_all=False,
                 samples = 256):
        
        assert mode in ['train', 'valid', 'test'], "mode must belong to one of these ['train', 'val', 'test']"
        
        f = h5py.File(path, 'r')
        self.mode = mode
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{nx}'
        self.samples = samples

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx):
        
        x = self.data['x'][idx]
        t = self.data['t'][idx]
        u_hr = torch.from_numpy(self.data[self.dataset][idx]).unsqueeze(0) # 1, T, L

        _, T, L = u_hr.shape

        if self.mode in ['train']:
            sample_lst = sorted(np.random.choice(T*L, self.samples, replace=False))
            hr_coord = make_coord([T,L])[sample_lst]
            hr_points = to_pixel_samples(u_hr)[1][sample_lst]
        else:
            hr_coord = make_coord([T,L])
            hr_points = to_pixel_samples(u_hr)[1]

        return_tensors = {
            't': t,
            'hr_frames': u_hr,
            'hr_points': hr_points, 
            'coords': hr_coord 
        }
        
        return return_tensors

class HDF5DatasetImplicitSlide(Dataset):
    
    def __init__(self, 
                 path,
                 nt,
                 nx,
                 time_slice,
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
        self.time_slice = time_slice
        self.n_total = nt

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]*(self.n_total-2*self.time_slice+1)

    def __getitem__(self, idx):
        
        nidx = idx // (self.n_total-2*self.time_slice+1)
        tidx = idx % (self.n_total-2*self.time_slice+1)

        x = self.data['x'][nidx]
        t = self.data['t'][nidx, tidx:tidx+2*self.time_slice]
        u_hr = torch.from_numpy(self.data[self.dataset][nidx, tidx:tidx+2*self.time_slice]).unsqueeze(1) # T, 1, L

        T, _, L = u_hr.shape

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
            
            u_hr_history = u_hr[:self.time_slice]
            u_hr_future = u_hr[self.time_slice:2*self.time_slice]
            hr_points_history = hr_points[:self.time_slice]
            hr_points_future = hr_points[self.time_slice:2*self.time_slice]
            
            return_tensors = {
            't': t,
            'sample_idx': sample_lst,
            'hr_frames_history': u_hr_history,
            'hr_frames_future': u_hr_future,
            'hr_points_history': hr_points_history,
            'hr_points_future': hr_points_future, 
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
                 nx: int,
                 dtype=torch.float32,
                 load_all: bool=False):
        """Initialize the dataset object.
        Args:
            path: path to dataset
            mode: [train, valid, test]
            nt: temporal resolution
            nx: spatial resolution
            shift: [fourier, linear]
            dtype: floating precision of data
            load_all: load all the data into memory
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.dtype = dtype
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{nx}'

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
        x = torch.from_numpy(self.data['x'][idx])
        t = torch.from_numpy(self.data['t'][idx])

        dx = torch.diff(x)[0]
        dt = torch.diff(t)[0]

        return u.float(), dx.float(), dt.float()

class HDF5DatasetSlide(Dataset):
    """
    Load samples of an PDE Dataset, get items according to PDE.
    """
    def __init__(self, 
                 path: str,
                 mode: str,
                 time_slice: int,
                 nt: int,
                 nx: int,
                 dtype=torch.float32,
                 load_all: bool=False):
        """Initialize the dataset object.
        Args:
            path: path to dataset
            mode: [train, valid, test]
            nt: temporal resolution
            nx: spatial resolution
            shift: [fourier, linear]
            dtype: floating precision of data
            load_all: load all the data into memory
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.dtype = dtype
        self.data = f[self.mode]
        self.time_slice = time_slice
        self.dataset = f'pde_{nt}-{nx}'
        self.n_total = nt

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]*(self.n_total-2*self.time_slice+1)

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
        nidx = idx // (self.n_total-2*self.time_slice+1)
        tidx = idx % (self.n_total-2*self.time_slice+1)
        
        x = torch.from_numpy(self.data['x'][nidx])
        t = torch.from_numpy(self.data['t'][nidx])
        dx = torch.diff(x)[0]
        dt = torch.diff(t)[0]
        
        u_history = torch.from_numpy(self.data[self.dataset][nidx, tidx:tidx+self.time_slice])
        u_future = torch.from_numpy(self.data[self.dataset][nidx, tidx+self.time_slice:tidx+2*self.time_slice])

        return u_history.float(), u_future.float(), dx.float(), dt.float()


class PDEDataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: str,
                 mode: str,
                 base_resolution: list=None,
                 super_resolution: list=None,
                 load_all: bool=False):
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE ('CE' or 'WE')
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.dtype = torch.float64
        self.data = f[self.mode]
        self.base_resolution = (250, 100) if base_resolution is None else base_resolution
        self.super_resolution = (250, 200) if super_resolution is None else super_resolution
        self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
        self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'

        ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
        ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
        assert (ratio_nt.is_integer())
        assert (ratio_nx.is_integer())
        self.ratio_nt = int(ratio_nt)
        self.ratio_nx = int(ratio_nx)

        self.nt = self.data[self.dataset_base].attrs['nt']
        self.dt = self.data[self.dataset_base].attrs['dt']
        self.dx = self.data[self.dataset_base].attrs['dx']
        self.x = self.data[self.dataset_base].attrs['x']
        self.tmin = self.data[self.dataset_base].attrs['tmin']
        self.tmax = self.data[self.dataset_base].attrs['tmax']

        if load_all:
            data = {self.dataset_super: self.data[self.dataset_super][:]}
            f.close()
            self.data = data


    def __len__(self):
        return self.data[self.dataset_super].shape[0]

    def __getitem__(self, idx: int):
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        if(f'{self.pde}' == 'CE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            
            left = u_super[..., -3:-1]
            right = u_super[..., 1:3]
            u_super_padded = torch.tensor(np.concatenate((left, u_super, right), -1)).float()
            weights = torch.tensor([[[[0.2]*5]]])
            u_super = F.conv1d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
            x = self.x

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['alpha'] = self.data['alpha'][idx]
            variables['beta'] = self.data['beta'][idx]
            variables['gamma'] = self.data['gamma'][idx]
            

            return u_super, self.dx, self.dt

        elif(f'{self.pde}' == 'WE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            # No padding is possible due to non-periodic boundary conditions
            weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            u_super = F.conv1d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # To match the downprojected trajectories, also coordinates need to be downprojected
            x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
            x = F.conv1d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['bc_left'] = self.data['bc_left'][idx]
            variables['bc_right'] = self.data['bc_right'][idx]
            variables['c'] = self.data['c'][idx]

            return u_super, self.dx, self.dt

        else:
            raise Exception("Wrong experiment")
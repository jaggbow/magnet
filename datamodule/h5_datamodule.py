from torch.utils.data import DataLoader

import pytorch_lightning as pl

from .dataset import *


class HDF5Datamodule(pl.LightningDataModule):
    def __init__(
        self, 
        name='h5_datamodule',
        train_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        val_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        test_path="/content/drive/MyDrive/MILA/snapshots.h5",
        nt_train=128,
        nx_train=256,
        nt_val=128,
        nx_val=256, 
        nt_test=256,
        nx_test=256,
        num_workers=2,
        batch_size = 32):
        super().__init__()

        self.save_hyperparameters()
        
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.nt_train = nt_train
        self.nx_train = nx_train
        self.nt_val = nt_val
        self.nx_val = nx_val
        self.nt_test = nt_test
        self.nx_test = nx_test
 
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        self.train_dataset = HDF5Dataset( 
                path=self.train_path,
                mode='train',
                nt=self.nt_train,
                nx=self.nx_train,
                dtype=torch.float32)

        self.val_dataset = HDF5Dataset( 
                path=self.val_path,
                mode='valid',
                nt=self.nt_val,
                nx=self.nx_val,
                dtype=torch.float32)
        
        self.test_dataset = HDF5Dataset( 
                path=self.test_path,
                mode='test',
                nt=self.nt_test,
                nx=self.nx_test,
                dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)


class HDF5DatamoduleImplicit(pl.LightningDataModule):
    def __init__(
        self, 
        name='h5_datamodule_implicit',
        train_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        val_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        test_path="/content/drive/MyDrive/MILA/snapshots.h5",
        nt_train=128,
        nx_train=256,
        nt_val=128,
        nx_val=256, 
        nt_test=256,
        nx_test=256,
        samples=32,
        sampling='uniform',
        num_workers=2,
        batch_size = 32):
        super().__init__()

        self.save_hyperparameters()
        
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.nt_train = nt_train
        self.nx_train = nx_train
        self.sampling = sampling
        self.nt_val = nt_val
        self.nx_val = nx_val
        self.nt_test = nt_test
        self.nx_test = nx_test
        self.samples = samples
 
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        self.train_dataset = HDF5DatasetImplicit( 
                path=self.train_path,
                mode='train',
                sampling=self.sampling,
                nt=self.nt_train,
                nx=self.nx_train,
                samples=self.samples)

        self.val_dataset = HDF5DatasetImplicit( 
                path=self.val_path,
                mode='valid',
                sampling=self.sampling,
                nt=self.nt_val,
                nx=self.nx_val,
                samples=self.samples)
        
        self.test_dataset = HDF5DatasetImplicit( 
                path=self.test_path,
                mode='test',
                sampling=self.sampling,
                nt=self.nt_test,
                nx=self.nx_test,
                samples=self.samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)


class HDF5DatamoduleGraph(pl.LightningDataModule):
    def __init__(
        self, 
        name='h5_datamodule_implicit',
        train_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        val_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        test_path="/content/drive/MyDrive/MILA/snapshots.h5",
        nt_train=128,
        nx_train=256,
        nt_val=128,
        nx_val=256, 
        nt_test=256,
        nx_test=256,
        in_timesteps=16,
        radius=2,
        num_workers=2,
        batch_size = 32):
        super().__init__()

        self.save_hyperparameters()
        
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.nt_train = nt_train
        self.nx_train = nx_train
        self.nt_val = nt_val
        self.nx_val = nx_val
        self.nt_test = nt_test
        self.nx_test = nx_test
        self.in_timesteps = in_timesteps
        self.radius = radius
 
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        self.train_dataset = HDF5DatasetGraph( 
                path=self.train_path,
                mode='train',
                nt=self.nt_train,
                nx=self.nx_train,
                in_timesteps=self.in_timesteps,
                radius=self.radius)

        self.val_dataset = HDF5DatasetGraph( 
                path=self.val_path,
                mode='valid',
                nt=self.nt_val,
                nx=self.nx_val,
                in_timesteps=self.in_timesteps,
                radius=self.radius)
        
        self.test_dataset = HDF5DatasetGraph( 
                path=self.test_path,
                mode='test',
                nt=self.nt_test,
                nx=self.nx_test,
                in_timesteps=self.in_timesteps,
                radius=self.radius)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)


class HDF5DatamoduleImplicitGNN(pl.LightningDataModule):
    def __init__(
        self, 
        name='h5_datamodule_implicit_gnn',
        train_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        val_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        test_path="/content/drive/MyDrive/MILA/snapshots.h5",
        nt_train=128,
        nx_train=256,
        nt_val=128,
        nx_val=256, 
        nt_test=256,
        nx_test=256,
        samples=32,
        sampling='uniform',
        num_workers=2,
        batch_size = 32):
        super().__init__()

        self.save_hyperparameters()
        
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.nt_train = nt_train
        self.nx_train = nx_train
        self.nt_val = nt_val
        self.nx_val = nx_val
        self.nt_test = nt_test
        self.nx_test = nx_test
        self.samples = samples
        self.sampling = sampling
 
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        self.train_dataset = HDF5DatasetImplicitGNN( 
                path=self.train_path,
                nt=self.nt_train,
                nx=self.nx_train,
                sampling=self.sampling,
                mode='train', 
                samples=self.samples)

        self.val_dataset = HDF5DatasetImplicitGNN( 
                path=self.val_path,
                nt=self.nt_val,
                nx=self.nx_val,
                sampling=self.sampling,
                mode='valid', 
                samples=self.samples)
        
        self.test_dataset = HDF5DatasetImplicitGNN( 
                path=self.test_path,
                nt=self.nt_test,
                nx=self.nx_test,
                sampling=self.sampling,
                mode='test', 
                samples=self.samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
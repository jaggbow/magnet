from cgitb import reset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from .dataset_2d import *


class HDF5Datamodule_2d(pl.LightningDataModule):
    def __init__(
        self, 
        name='h5_datamodule_2d',
        train_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        val_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        test_path="/content/drive/MyDrive/MILA/snapshots.h5",
        nt_train=128,
        res_train=256,
        nt_val=128,
        res_val=256, 
        nt_test=256,
        res_test=256,
        num_workers=2,
        batch_size=32):
        super().__init__()

        self.save_hyperparameters()
        
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.nt_train = nt_train
        self.res_train = res_train
        self.nt_val = nt_val
        self.res_val = res_val
        self.nt_test = nt_test
        self.res_test = res_test
 
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        self.train_dataset = HDF5Dataset( 
                path=self.train_path,
                mode='train',
                nt=self.nt_train,
                res=self.res_train,
                dtype=torch.float32)

        self.val_dataset = HDF5Dataset( 
                path=self.val_path,
                mode='test',
                nt=self.nt_val,
                res=self.res_val,
                dtype=torch.float32)
        
        self.test_dataset = HDF5Dataset( 
                path=self.test_path,
                mode='test',
                nt=self.nt_test,
                res=self.res_test,
                dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)


class HDF5DatamoduleImplicit_2d(pl.LightningDataModule):
    def __init__(
        self, 
        name='h5_datamodule_implicit_2d',
        train_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        val_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        test_path="/content/drive/MyDrive/MILA/snapshots.h5",
        nt_train=128,
        res_train=256,
        nt_val=128,
        res_val=256, 
        nt_test=256,
        res_test=256,
        samples=32,
        num_workers=2,
        batch_size = 32):
        super().__init__()

        self.save_hyperparameters()
        
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.nt_train = nt_train
        self.res_train = res_train
        self.nt_val = nt_val
        self.res_val = res_val
        self.nt_test = nt_test
        self.res_test = res_test
        self.samples = samples
 
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        self.train_dataset = HDF5DatasetImplicit_2d( 
                path=self.train_path,
                mode='train',
                nt=self.nt_train,
                res=self.res_train,
                samples=self.samples)

        self.val_dataset = HDF5DatasetImplicit_2d( 
                path=self.val_path,
                mode='test',
                nt=self.nt_val,
                res=self.res_val,
                samples=self.samples)
        
        self.test_dataset = HDF5DatasetImplicit_2d( 
                path=self.test_path,
                mode='test',
                nt=self.nt_test,
                res=self.res_test,
                samples=self.samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)


class HDF5DatamoduleGraph_2d(pl.LightningDataModule):
    def __init__(
        self, 
        name='h5_datamodule_graph_2d',
        train_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        val_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        test_path="/content/drive/MyDrive/MILA/snapshots.h5",
        nt_train=128,
        res_train=256,
        nt_val=128,
        res_val=256, 
        nt_test=256,
        res_test=256,
        train_regular=True,
        val_regular=True,
        test_regular=True,
        num_workers=2,
        batch_size=32):
        super().__init__()

        self.save_hyperparameters()
        
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.nt_train = nt_train
        self.res_train = res_train
        self.nt_val = nt_val
        self.res_val = res_val
        self.nt_test = nt_test
        self.res_test = res_test
        self.train_regular = train_regular
        self.val_regular = val_regular
        self.test_regular = test_regular
 
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        self.train_dataset = HDF5DatasetGraph_2d( 
                path=self.train_path,
                mode='train',
                regular=self.train_regular,
                nt=self.nt_train,
                res=self.res_train)

        self.val_dataset = HDF5DatasetGraph_2d( 
                path=self.val_path,
                mode='test',
                regular=self.val_regular,
                nt=self.nt_val,
                res=self.res_val)
        
        self.test_dataset = HDF5DatasetGraph_2d( 
                path=self.test_path,
                mode='test',
                regular=self.test_regular,
                nt=self.nt_test,
                res=self.res_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)


class HDF5DatamoduleImplicitGNN_2d(pl.LightningDataModule):
    def __init__(
        self, 
        name='h5_datamodule_implicit_gnn',
        train_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        val_path="/content/drive/MyDrive/MILA/snapshots.h5", 
        test_path="/content/drive/MyDrive/MILA/snapshots.h5",
        nt_train=128,
        res_train=256,
        nt_val=128,
        res_val=256, 
        nt_test=256,
        res_test=256,
        train_regular=False,
        val_regular=True,
        test_regular=True,
        samples=32,
        num_workers=2,
        batch_size = 32):
        super().__init__()

        self.save_hyperparameters()
        
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.nt_train = nt_train
        self.res_train = res_train
        self.nt_val = nt_val
        self.res_val = res_val
        self.nt_test = nt_test
        self.res_test = res_test
        self.samples = samples
        self.train_regular = train_regular
        self.val_regular = val_regular
        self.test_regular = test_regular
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        self.train_dataset = HDF5DatasetImplicitGNN_2d( 
                path=self.train_path,
                nt=self.nt_train,
                res=self.res_train,
                mode='train', 
                regular=self.train_regular,
                samples=self.samples)

        self.val_dataset = HDF5DatasetImplicitGNN_2d( 
                path=self.val_path,
                nt=self.nt_val,
                res=self.res_val,
                mode='test', 
                regular=self.val_regular,
                samples=self.samples)
        
        self.test_dataset = HDF5DatasetImplicitGNN_2d( 
                path=self.test_path,
                nt=self.nt_test,
                res=self.res_test,
                mode='test', 
                regular=self.test_regular,
                samples=self.samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
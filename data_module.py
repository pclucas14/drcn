import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from torch.utils.data import Sampler, ConcatDataset, DataLoader
from datasets import get_dataset
from typing import Optional

class ImageDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        # get path to data
        self.args = args
        self.location = args.data_location

    def setup(self, stage=None):
        src_ds_class = get_dataset(self.args.source_dataset)
        src_tr = src_ds_class(self.location, "train", self.args.img_size)
        src_val = src_ds_class(self.location, "val", self.args.img_size)
        src_te = src_ds_class(self.location, "test", self.args.img_size)

        datasets = [
            DatasetWrapper(ds, True) for ds in [src_tr, src_val, src_te]
        ]

        tgt_ds_class = get_dataset(self.args.target_dataset)
        tgt_tr = tgt_ds_class(self.location, "train", self.args.img_size)
        tgt_val = tgt_ds_class(self.location, "val", self.args.img_size)
        tgt_te = tgt_ds_class(self.location, "test", self.args.img_size)
                
        tgt_datasets = [
            DatasetWrapper(ds, False) for ds in [tgt_tr, tgt_val, tgt_te]
        ]

        datasets = [
            ConcatDataset([src, tgt]) for src, tgt, in zip(datasets, tgt_datasets)
        ]

        self.train_ds, self.val_ds, self.test_ds = datasets

    def train_dataloader(self):
        sampler = None
        if self.args.no_mix_src_and_tgt:
            sampler = SeparateShuffleSampler(self.train_ds)

        return DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            sampler=sampler,
            shuffle=sampler is None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.batch_size * 2,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.batch_size * 2,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )


class DatasetWrapper(torch.utils.data.Dataset):
    """
    Dataset wrapper that also outputs from which domain the data comes from
    """

    def __init__(self, dataset, is_source):
        self.dataset = dataset
        self.is_source = is_source

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return *self.dataset[idx], self.is_source


class SeparateShuffleSampler(Sampler):
    """
    Sampler that shuffles both source and target datasets, but first presents
    all the data from source, and proceeds with the data from target
    """

    def __init__(self, concat_dataset):
        self.dataset = concat_dataset
        self.source_ds, self.target_ds = concat_dataset.datasets
        assert self.source_ds.is_source and not self.target_ds.is_source

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # what's the limit ?
        src_len, tgt_len = len(self.source_ds), len(self.target_ds)
        source_shuffled = torch.randperm(src_len)
        target_shuffled = torch.randperm(tgt_len) + src_len
        indices = torch.cat((source_shuffled, target_shuffled))
        return iter(indices) 
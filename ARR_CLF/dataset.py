# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:12:10 2019

@author: xuxinzi
"""

import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset,WeightedRandomSampler

from torchvision import transforms




class Dataset():

    def __init__(self,  train_data, train_label =None, transform= None):
        super(Dataset,self).__init__()
        self.transform = transform

        self.ecg_img = train_data            
        self.label = train_label
 
    def __len__(self):
        return self.ecg_img.shape[0]

    
    def __getitem__(self, item):
        ecg = self.ecg_img[item]
        # print(ecg.shape)
        ecg = np.expand_dims(ecg, axis = 1) # 1dcnn
        # print(ecg.shape)
        
        if self.transform is not None:
            # print('ecg', ecg.shape)
            ecg = self.transform(ecg)

        if self.label is not None:
            label = self.label[item]
            return ecg, label
        else:
            return ecg
            
# 

def get_train_loader(batch_size,train_data, train_label):
    """
    Utility function for loading and returning a multi-process
    train iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: train set iterator.
    """


    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])    

    train_dataset = Dataset(train_data, train_label, 
                                    transform= train_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True,
                                                num_workers=8, pin_memory=True)            
    return train_loader



def get_test_loader(batch_size, test_data, test_label):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])    
    test_dataset = Dataset(test_data, test_label,transform= test_transform)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            num_workers=8, pin_memory=True)

    return test_loader


def get_infer_loader(batch_size,infer_data):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])    
    infer_dataset = Dataset(infer_data, transform= test_transform)
    infer_loader = torch.utils.data.DataLoader(dataset = infer_dataset,
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            num_workers=8, pin_memory=False)

    return infer_loader
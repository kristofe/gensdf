#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from . import base 

import pandas as pd 
import numpy as np
import csv

class LabeledDS(base.Dataset):

    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes 
        samples_per_mesh=16000,
        pc_size=1024,
        max_meshes=1000
    ):

        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.gt_files = self.get_instance_filenames(data_source, split_file)
        if(len(self.gt_files) > max_meshes):
            self.gt_files = self.gt_files[0:max_meshes]

    def __getitem__(self, idx): 
        
        pc, sdf_xyz, sdf_gt =  self.labeled_sampling_npz(self.gt_files[idx], self.samples_per_mesh, self.pc_size)

        data_dict = {
                    "sdf_xyz":sdf_xyz.float().squeeze(),
                    "gt_sdf":sdf_gt.float().squeeze(), 
                    "point_cloud":pc.float().squeeze(),
                    "indices":idx                
                    }

        return data_dict

    def __len__(self):
        return len(self.gt_files)



    
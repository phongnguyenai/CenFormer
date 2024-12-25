import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import logging
import random
import torch.nn.functional as F
from torch_geometric.data import Data
import sys
sys.path.append("../")
from utils import misc

@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.mode = config.MODE
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        gt = Data(pos=data.squeeze())

        if self.subset == "train":
            partial, _ = misc.seprate_point_cloud(data.unsqueeze(0).cuda(), self.npoints, [int(self.npoints * 1/4) , int(self.npoints * 3/4)], fixed_points = None)
            partial = Data(pos=partial.squeeze())
            return sample['taxonomy_id'], sample['model_id'], partial, gt
        elif self.subset == "test":
            if self.mode == "hard":
                partial, _ = misc.seprate_point_cloud(data.unsqueeze(0).cuda(), self.npoints,  int(self.npoints * 3/4), fixed_points = None)
            elif self.mode =="medium":
                partial, _ = misc.seprate_point_cloud(data.unsqueeze(0).cuda(), self.npoints,  int(self.npoints * 1/2), fixed_points = None)
            elif self.mode =="easy":
                partial, _ = misc.seprate_point_cloud(data.unsqueeze(0).cuda(), self.npoints,  int(self.npoints * 1/4), fixed_points = None)
            elif self.mode =="f_score":
                partial, _ = misc.seprate_point_cloud(data.unsqueeze(0).cuda(), self.npoints, [int(self.npoints * 1/4) , int(self.npoints * 3/4)], fixed_points = None)
            
            partial = misc.fps(partial, 2048)
            partial = Data(pos=partial.squeeze())

            return sample['taxonomy_id'], sample['model_id'], partial, gt
        
        # elif self.subset == "test":
        #     lst_partial = []
        #     choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
        #                     torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
        #     for item in choice:
        #         if self.mode == "hard":
        #             partial, _ = misc.seprate_point_cloud(data.unsqueeze(0).cuda(), self.npoints, int(self.npoints*3/4), fixed_points = item)
        #         elif self.mode == "medium":
        #             partial, _ = misc.seprate_point_cloud(data.unsqueeze(0).cuda(), self.npoints, int(self.npoints*1/2), fixed_points = item)
        #         elif self.mode == "easy":
        #             partial, _ = misc.seprate_point_cloud(data.unsqueeze(0).cuda(), self.npoints, int(self.npoints*1/4), fixed_points = item)

        #         partial = misc.fps(partial, 2048)
        #         partial = Data(pos=partial.squeeze())
        #         lst_partial.append(partial)

            return sample['taxonomy_id'], sample['model_id'], lst_partial, gt

    def __len__(self):
        return len(self.file_list)

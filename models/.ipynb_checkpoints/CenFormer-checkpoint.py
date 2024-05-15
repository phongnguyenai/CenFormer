'''References
https://github.com/chenzhik/AnchorFormer
'''

import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.expansion_penalty.expansion_penalty_module import expansionPenaltyModule
from .Centroids import CGB_CDB
from .FPG import FPG

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc    

class CenFormer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.dim = config.dim
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        self.sparse_expansion_lambda = config.sparse_expansion_lambda
        self.dense_expansion_lambda = config.dense_expansion_lambda
        self.up_ratio = self.num_pred//self.num_query
        
        self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5)
        
        self.encoder = CGB_CDB(num_query=self.num_query, dim=self.dim)
        self.decoder = FPG(self.dim, step = self.fold_step, hidden_dim = 256) 

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map_global = nn.Linear(1024, self.dim)
        self.reduce_map_local = nn.Linear(self.dim + 3, self.dim)
        
        self.include_input = False
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        self.penalty_func = expansionPenaltyModule()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine
    
    def get_penalty(self):
        dist, _, mean_mst_dis = self.penalty_func(self.pred_coarse_point, 16, self.sparse_expansion_lambda)
        dist_dense, _, mean_mst_dis = self.penalty_func(self.pred_dense_point, 64, self.dense_expansion_lambda)
        loss_mst = torch.mean(dist) 
        loss_mst_fine = torch.mean(dist_dense) 
        return loss_mst, loss_mst_fine
    
    def forward(self, pxo):
        q, coarse_point_cloud = self.encoder(pxo) # B M C and B M 3

        self.pred_coarse_point = coarse_point_cloud
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024
        
        rebuild_feature = torch.cat([global_feature.unsqueeze(-2).expand(-1, M, -1), q, coarse_point_cloud], dim=-1)  # B M 1027 + C
        
        global_feature = rebuild_feature[:,:,:1024].reshape(B*M,1024)
        local_feature = rebuild_feature[:,:,1024:].reshape(B*M,self.dim+3)
        
        global_feature = self.reduce_map_global(global_feature)
        local_feature = self.reduce_map_local(local_feature)

        relative_xyz = self.decoder(global_feature, local_feature).reshape(B, M, 3, -1)    # B M 3 S
        
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3
        
        self.pred_dense_point = rebuild_points

        xyz = pxo[0].reshape(pxo[2].shape[0], -1, 3)
        inp_sparse = fps(xyz, self.num_query)
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        if self.include_input: rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()

        ret = (coarse_point_cloud, rebuild_points)
        return ret

'''References
https://github.com/chenzhik/AnchorFormer
https://github.com/POSTECH-CVLab/point-transformer
'''

import torch
import torch.nn.functional as F
from torch import nn
from lib.pointops.functions import pointops
from timm.models.layers import DropPath,trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
class SelfFusionLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
        
class CrossFusion(nn.Module):
    def __init__(self, dim, out_dim, num_heads=1, qkv_bias=False, qk_scale=1, attn_drop=0., proj_drop=0., aggregate_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Identity() # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Identity() # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.x_map = nn.Identity() # nn.Linear(aggregate_dim, 1)
        
    def forward(self, q, k, v):
        B, N, _ = q.shape
        C = self.out_dim
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, NK, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, 3)

        x = self.x_map(x)

        return x
        
class PointAggregation(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
            self.attention = SelfAttention(3+in_planes)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.int(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(self.attention(x)).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]

class SelfFusion(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(SelfFusion, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = SelfFusionLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

def PA_SF(in_plane, out_plane, num_blocks, share_planes=8, stride=1, nsample=16):
    block = nn.Sequential()
    block.add_module(name='PointAggregation', module=PointAggregation(in_plane, out_plane, stride, nsample))
    
    for i in range(1, num_blocks):
        block.add_module(name=f'SelfFusion_{i}', module=SelfFusion(out_plane, out_plane, share_planes, nsample=nsample))

    return block

class SubFold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()
        self.in_channel = in_channel
        self.step = step
        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x, c):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = c.to(x.device) # b 3 n2

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2
        
class CGB(nn.Module):
    def __init__(self, func, num_point, in_plane, out_plane, num_blocks, share_planes=8, stride=1, nsample=16, num_pred=16):
        super(CGB, self).__init__()
        self.func = func
        self.pa_sf = PA_SF(in_plane, out_plane, num_blocks, share_planes, stride, nsample)
        self.num_point = num_point
        self.generate_feature = nn.Sequential(
            nn.Conv1d(self.num_point, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, num_pred, 1)
        )
        self.norm_q = nn.Identity() # norm_layer(dim)
        self.norm_k = nn.Identity () # norm_layer(dim)
        self.cf = CrossFusion(out_plane, out_plane, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., aggregate_dim=16)

        self.fold_step = int(pow(num_pred, 0.5) + 0.5)
        self.generate_centroids = SubFold(out_plane, step = self.fold_step, hidden_dim = out_plane // 2)
        
    def forward(self, pxo):
        if self.func == 'generate_features':
            p, x, o = pxo
            p_2, x_2, o_2 = self.pa_sf(pxo)
    
            return p_2, x_2, o_2
            
        elif self.func == 'generate_centroids':
            # From pos to normal tensor
            p, x, o = pxo
            b = o.shape[0] # batch size
            p = p.reshape(b, -1, p.shape[1])
            x = x.reshape(b, -1, x.shape[1])
            
            # Locally preserve non-misssing part
            p_2, x_2, o_2 = self.pa_sf(pxo)
            p_2 = p_2.reshape(b, -1, p_2.shape[1])
            x_2 = x_2.reshape(b, -1, x_2.shape[1])

            # Offset to predict missing part
            global_x = torch.max(x_2, dim=1, keepdim=False)[0] # B dim
            diff_x = global_x.unsqueeze(1).repeat(1,self.num_point,1) - x
            x_3 = self.generate_feature(diff_x)

            norm_k = self.norm_k(x) # B N dim
            norm_q = self.norm_q(x_3) # B L dim
            p_3 = self.cf(q=norm_q, k=norm_k, v=p)
            p_3 = self.generate_centroids(global_x, p_3.transpose(1,2)).transpose(1,2)
            
            x = torch.cat([x, x_2, x_3], dim=1)        
            p = torch.cat([p, p_2, p_3], dim=1) # coor: B N 3 -> B N+L 3

            for i in range(o.shape[0]):
                o[i] = o[i] + (p_2.shape[1] + p_3.shape[1])*(i+1)

            return p.reshape(-1, p.shape[-1]), x.reshape(-1, x.shape[-1]), o

class FeatureDispersion(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, q, v):
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        q = q + self.drop_path(q_2)
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q
        
class CGB_CDB(nn.Module):
    def __init__(self, num_inpoint=2048, dim=256, num_blocks=2, stride_feature=16, stride_centroid=8, share_planes=8, num_query=256, num_heads=6, num_enc=4, num_dec=4):
        super().__init__()

        self.feature_block = CGB(func='generate_features', num_point=num_inpoint, in_plane=3, out_plane=dim, num_blocks=num_blocks, share_planes=8, stride=stride_feature, nsample=16, num_pred=16).cuda()
        
        # centroids generation
        lst = []
        num_point = 128
        stride_factor = 2
        for i in range(num_enc):
            lst.append(CGB(func='generate_centroids', num_point=num_point, in_plane=dim, out_plane=dim, num_blocks=num_blocks, share_planes=8, stride=stride_centroid, nsample=8, num_pred=16))
            num_point = int(num_point + 16 +16)
            stride_centroid+=stride_factor
            
        self.centroid_blocks = torch.nn.ModuleList(lst)

        self.increase_dim = nn.Sequential(
            nn.Conv1d(dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * (num_query-128))
        )
        
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, dim, 1)
        )

        self.fd = nn.ModuleList([
            FeatureDispersion(
                dim=dim, num_heads=num_heads, mlp_ratio=2., qkv_bias=None, qk_scale=None,
                drop=0., attn_drop=0.)
            for i in range(num_dec)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, inpc):
        enc = self.feature_block(inpc)
        
        for i, blk in enumerate(self.centroid_blocks): 
            enc = blk(enc)

        coor, x, o = enc
        
        b = o.shape[0] # batch size
        coor = coor.reshape(b, -1, coor.shape[1])
        x = x.reshape(b, -1, x.shape[1])
        
        # Point Dispersion
        global_feature = self.increase_dim(x.transpose(1,2))  
        global_feature = torch.max(global_feature, dim=-1)[0] 
        
        coarse_relative = torch.cat([torch.zeros([b, 128, 3]).to(x.device), self.coarse_pred(global_feature).reshape(b, -1, 3)], dim=1)  
        coarse_point = coarse_relative + coor
        
        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, coarse_point.size(1), -1), 
            coarse_point], dim=-1) # B M C+3 
        
        q = self.mlp_query(query_feature.transpose(1,2)).transpose(1,2) # B M C

        # Feature Dispersion
        for i, blk in enumerate(self.fd): 
            q = blk(q, x)
            
        return q, coarse_point
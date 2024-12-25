from __future__ import print_function
import os
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from models.CenFormer import CenFormer
import argparse
from datasets import PCNDataset
import open3d as o3d

class Config( object ):
    def __init__( self ):
        self.myAttr= None

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_o3d(pc, fname):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(fname, pcd)

def predict(data, cate):
    path = f"output/{cate}"
    if not os.path.exists(path):
        os.makedirs(path)
    gt = data[3].to(device)
    partial = data[2].to(device)
    p, x, o = partial.pos, partial.pos, partial.ptr[1:]
    b = o.shape[0]

    save_o3d(partial.pos.detach().cpu().squeeze().numpy(), f"{path}/partial.ply")
    save_o3d(gt.pos.detach().cpu().squeeze().numpy(), f"{path}/gt.ply")
    
    with torch.no_grad():
        ret = model([p, x, o])
        dense_points = ret[1]

        xyz = dense_points[0].detach().cpu().squeeze().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.visualization.draw_geometries([pcd])
        save_o3d(xyz, f"{path}/prediction.ply")
        print("Saved in " + f"{path}/prediction.ply")
        

def run(cate, min, max):
    print("Predicting...")
    count = 1
    value =  random.randint(min, max)
    
    for data in test_dataloader:
        if count == value:
            predict(data, cate)
            break
        count += 1

seed_everything()

# Test cofig
test_config = Config()
test_config.subset = "test"
test_config.PARTIAL_POINTS_PATH = f"PCN/{test_config.subset}/partial"
test_config.COMPLETE_POINTS_PATH = f"PCN/{test_config.subset}/complete"
test_config.CATEGORY_FILE_PATH = "PCN/PCN.json"
test_config.N_POINTS = 16384
test_config.CARS = False

test_dataset = PCNDataset.PCN(test_config)

test_dataloader = DataLoader(test_dataset, batch_size=15,
                    shuffle=False)

parser = argparse.ArgumentParser()
parser.add_argument("--cate", help="categories", default="airplane")
parser.add_argument("--pretrained", help="path to pretrained model", default="pretrained/PCN/best.pt")

args = parser.parse_args()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_config = Config()
model_config.num_pred = 16384
model_config.num_query = 256
model_config.dim = 384
model_config.sparse_expansion_lambda = 0.5
model_config.dense_expansion_lambda = 1.2
model = CenFormer(model_config).cuda()
model.load_state_dict(torch.load(args.pretrained))

if args.cate == 'airplane':
    run(args.cate, 1, 10)
elif args.cate == 'cabinet':
    run(args.cate, 11, 20)
elif args.cate == 'car':
    run(args.cate, 21, 30)
elif args.cate == 'chair':
    run(args.cate, 31, 40)
elif args.cate == 'lamp':
    run(args.cate, 41, 50)
elif args.cate == 'sofa':
    run(args.cate, 51, 60)
elif args.cate == 'table':
    run(args.cate, 61, 70)
elif args.cate == 'watercraft':
    run(args.cate, 71, 80)

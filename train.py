from __future__ import print_function
import os
import random
import numpy as np
import torch
from torch_geometric.data import DataLoader
from datasets import PCNDataset
import argparse
import sys
from torch.optim.lr_scheduler import StepLR
from models.CenFormer import CenFormer
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", help="pth file", default="")
parser.add_argument("--car", help="True or False", default=False)
parser.add_argument("--batch-size", help="Batch size", default=8)
parser.add_argument("--model-name", help="Name of the method", default="CenFormer")
parser.add_argument("--epoch", help="Epoch", default=401)
parser.add_argument("--num-pred", help="Number of output point cloud", default=16384)

args = parser.parse_args()

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

def train(epoch):
    model.train()
    train_loss = 0
    count = 0
    for data in train_dataloader:
        gt = data[3].to(device)
        partial = data[2].to(device)
        
        p, x, o = partial.pos, partial.pos, partial.ptr[1:]
        b = o.shape[0]
        
        # Prediction
        optimizer.zero_grad()

        ret = model([p, x, o])
        sparse_loss, dense_loss = model.get_loss(ret, gt.pos.reshape(b,int(args.num_pred),3))
        sparse_penalty, dense_penalty = model.get_penalty()
        sparse_loss = 1.0 * sparse_loss
        dense_loss = 1.0 * dense_loss
        
        sparse_penalty = 0.1 * sparse_penalty
        dense_penalty = 0.05 * dense_penalty
        loss = sparse_loss + dense_loss + sparse_penalty + dense_penalty
        
        torch.cuda.empty_cache()
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
        count +=1
        print(f"{model_name} - Training epoch {epoch}: {int(count/len(train_dataloader)*100)}%", end='\r')
        sys.stdout.flush()
        
    return train_loss / len(train_dataset)

def evaluation():
    model.eval()
    val_loss = 0
    count = 0
    
    for data in val_dataloader:      
        gt = data[3].to(device)
        partial = data[2].to(device)
        
        p, x, o = partial.pos, partial.pos, partial.ptr[1:]
        b = o.shape[0]
        
        # Prediction
        with torch.no_grad():
            
            ret = model([p, x, o])
            
            dense_points = ret[1]
            loss =  ChamferDisL2(dense_points, gt.pos.reshape(b, int(args.num_pred),3))
            val_loss += loss * b
            
        count +=1
        print(f"{model_name} - Validating epoch {epoch}: {int(count/len(val_dataloader)*100)}%", end='\r')
        sys.stdout.flush()
            
    return val_loss/len(val_dataset)

seed_everything()

# Train cofig
train_config = Config()
train_config.subset = "train"
train_config.PARTIAL_POINTS_PATH = f"PCN/{train_config.subset}/partial"
train_config.COMPLETE_POINTS_PATH = f"PCN/{train_config.subset}/complete"
train_config.CATEGORY_FILE_PATH = "PCN/PCN.json"
train_config.N_POINTS = int(args.num_pred)
train_config.CARS = False

# Valid cofig
val_config = Config()
val_config.subset = "val"
val_config.PARTIAL_POINTS_PATH = f"PCN/{val_config.subset}/partial"
val_config.COMPLETE_POINTS_PATH = f"PCN/{val_config.subset}/complete"
val_config.CATEGORY_FILE_PATH = "PCN/PCN.json"
val_config.N_POINTS = int(args.num_pred)
val_config.CARS = False

# Dataset
train_dataset = PCNDataset.PCN(train_config)
val_dataset = PCNDataset.PCN(val_config)

# Dataloader
batch_size = int(args.batch_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                    shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                    shuffle=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_config = Config()
model_config.num_pred = 16384
model_config.num_query = 256
model_config.dim = 384
model_config.sparse_expansion_lambda = 0.5
model_config.dense_expansion_lambda = 1.2

model = CenFormer(model_config).cuda()

if args.pretrained != "":
    model.load_state_dict(torch.load(args.pretrained))

# optimizer
ChamferDisL2 = ChamferDistanceL2()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Model name
model_name = args.model_name

# torch.cuda.empty_cache()
torch.set_num_threads(24)

if not os.path.exists(f"pretrained/PCN/{model_name}"):
    os.makedirs(f"pretrained/PCN/{model_name}")
    
scheduler = StepLR(optimizer, step_size=40, gamma=0.7)
# Start training
for epoch in range(int(args.epoch)):
# for epoch in range(401, int(args.epoch)):
    
    train_loss = train(epoch)
    print('Epoch {:03d}, Training loss: {:.10f}'.format(epoch, train_loss))
    with open(f'./pretrained/PCN/{model_name}/{model_name}_train_loss.txt', 'a') as f:
        f.write('Epoch {:03d}, Training loss: {:.10f} \n'.format(epoch, train_loss))
            
    if epoch%50==0:
        eval_loss = evaluation()
        print('Epoch {:03d}, Evaluation loss: {:.10f}'.format(epoch, eval_loss))

        str_loss = "Epoch {:03d} - Training loss: {:.10f} - Validation loss: {:.10f} \n".format(epoch, train_loss, eval_loss)
        with open(f'./pretrained/PCN/{model_name}/{model_name}_train_loss.txt', 'a') as f:
            f.write(str_loss)
            torch.save(model.state_dict(),'./pretrained/PCN/{}/{}_epoch_{}_train_{:.10f}_val_{:.10f}.pt'.format(model_name, model_name, epoch,train_loss,eval_loss))
            
    scheduler.step()

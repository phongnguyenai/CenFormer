# CenFormer: Transformer-based Network from Centroid Generation for Point Cloud Completion

![plot](./figs/overview.jpg)

## 1. Revision 1

We have updated the code and pretrained models for KITTI dataset and ShapeNet dataset. All the pre-trained models can be found [here](https://uowmailedu-my.sharepoint.com/:u:/g/personal/ttpn997_uowmail_edu_au/IQBu3Z4DmFfGRLSpq7MXT4T7AW6rL72GaIb4IdoZoeUHSBE?e=PxKKUo).

To download datasets, please check PoinTr (https://github.com/yuxumin/PoinTr/tree/master).

## 2. Installation

We provide instructions for creating a conda environment for training and predicting. (Note that we use CUDA Version 11.8).

```
sh ./env.sh
```

## 3. Sample code on PCN dataset

### 3.1. Prediction

To make predictions, follow these steps:

1. Download the test dataset from [this link](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/supplementary-papers/CenFormer/test.tar.gz?csf=1&web=1&e=RXWgkj). Extract it to the folder `./PCN/test/`.

2. Download the pretrained weight (**pretrained/PCN/best.pt**) from [this link](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/supplementary-papers/CenFormer/best.pt?csf=1&web=1&e=2rG5pI). Then put the pretrained weight in the folder `./pretrained/PCN/`.


3. Run the following command to perform predictions:

    ```
    python predict.py --cate airplane --pretrained pretrained/PCN/best.pt
    ```

    Parameters:

   ```
   --cate (str) categories, such as airplane, cabinet, car, chair, lamp, sofa, table, watercraft.
   
   --pretrained (str) path to the pretrained model.
   ```

![plot](./figs/PCN.png)

### 3.2. Training

To train the model, you need to follow these steps:

1. Download the validation dataset from [this link](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/supplementary-papers/CenFormer/val.tar.gz?csf=1&web=1&e=NLNvxX).

2. Download the training dataset from [this link](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/supplementary-papers/CenFormer/train.tar.gz?csf=1&web=1&e=M05LN7).

3. After downloading, extract the validation and training dataset files to the folders `./PCN/train/` and `./PCN/val/`, respectively.

4. To initiate the training process, execute the following command:

   ```
   python train.py --car False --batch-size 8 --model-name CenFormer --epoch 401 --num-pred 16384
   ```

   Parameters:
    
   ```
   --pretrained (str) path to the pretrained model.
    
   --car (bool) only use the CAR category for training.
    
   --batch-size (int) batch size.
    
   --model-name (str) name of the model.
    
   --epoch (int) number of epochs.
    
   --num-pred (int) number of the points in the final point cloud.
   ```
    
## References

Thank you [AnchorFormer](https://github.com/chenzhik/AnchorFormer), [PointTransformer](https://github.com/POSTECH-CVLab/point-transformer), and [PoinTr](https://github.com/yuxumin/PoinTr) for your wonderful code.

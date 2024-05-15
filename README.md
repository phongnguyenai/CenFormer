# CenFormer: Transformer-based Network from Centroid Generation for Point Cloud Completion

## Installation

We provide instructions for creating a conda environment for training and predicting. (Note that we use CUDA Version 11.8).

```
sh ./env.sh
```

## Prediction

To make predictions, follow these steps:

1. Download the test dataset from [this link](https://uowmailedu-my.sharepoint.com/:u:/r/personal/ttpn997_uowmail_edu_au/Documents/dataset/ATT-Net/test.tar.gz?csf=1&web=1&e=Sn6rpK). Extract it to the folder `./PCN/test/`.

2. Download the pretrained weight (**best.pt**) from [this link](https://uowmailedu-my.sharepoint.com/:u:/g/personal/ttpn997_uowmail_edu_au/EWcJcmV2ruNKsNfQiGLIMUgBvOL1s5fa6zA7rcIIrN33Ug?e=lcbOyE). Then put the pretrained weight in the folder `./pretrained/CenFormer/`.


3. Run the following command to perform predictions:

    ```bash
    python predict.py --cate <category> --pretrained <path to the pretrained model>
    ```

## Training

To train the model, you need to follow these steps:

1. Download the validation dataset from [this link](https://uowmailedu-my.sharepoint.com/:u:/g/personal/ttpn997_uowmail_edu_au/EbxYcKtV_ahOpaAvq-A-9ZwBOqabr_5nddl7mWwhWJJ_Rw?e=FSiE7A).

2. Download the training dataset from [this link](https://uowmailedu-my.sharepoint.com/:u:/g/personal/ttpn997_uowmail_edu_au/EeffEPj7HgpGhkGQVshxqWwBRz6bGUjLmirj79GgFflyCA?e=HhemQE).

3. After downloading, extract the validation and training dataset files to the folders `./PCN/train/` and `./PCN/val/`, respectively.

4. To initiate the training process, execute the following command:

    ```bash
    python train.py --pretrained <path to the pretrained model> --car <Only use CAR category for training> --batch-size <batch size> --model-name <name of the model> --epoch <Number of epochs> --num-pred <number of predicted points>
    ```

## References

[1] https://github.com/chenzhik/AnchorFormer

[2] https://github.com/POSTECH-CVLab/point-transformer


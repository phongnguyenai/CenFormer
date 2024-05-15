conda create -n CenFormer python=3.10
conda activate CenFormer
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip3 install timm
cd Pointnet2_Pytorch
python setup.py install
cd lib/pointops
python setup.py install
# Getting Started
## Installation
This repo is tested with Ubuntu 18.04, python==3.7, pytorch==1.4.0 and cuda==10.1

```
docker pull pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
```

Install PyTorch and other dependencies:

```console
conda create -n monoflex python=3.7
conda activate monoflex
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

Build DCNv2 and the project
```console
cd models/backbone/DCNv2
. make.sh
cd ../../..
python setup develop
```

Then modify the paths in config/paths_catalog.py according to your data path (default: ../datasets/kitti).

## Training

Training model in supervised model.
```console
python tools/plain_train_net.py --batch_size 8 --backbone dla34 --config runs/monoflex.yaml --output output/exp
```

Training model in semi-supervised model.
```console
python tools/plain_train_net.py --batch_size 8 --backbone dla34 --config runs/monoflex.yaml --mix_teaching True --ckpt <path-to-previous-teacher-model> --output output/exp
```

## Inference 
```console
python tools/plain_train_net.py --config runs/monoflex_test.yaml --ckpt YOUR_CKPT  --output <path-to-save-results> --eval
```
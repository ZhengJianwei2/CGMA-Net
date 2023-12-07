# CGMA-Net

## Papers
* CGMA-Net: CGMA-Net: Cross-Level Guidance and Multi-scale Aggregation Network for Polyp Segmentation
## 1. Environment setup
This code has been tested on on the workstation with NVIDIA RTX 3090 GPU with 24GB of video memory, Python 3.7, pytorch 1.11.0, CUDA 11.3, cuDNN 8.2. Please install related libraries before running this code:

    pip install -r requirements.txt
## 2. Download the datesets:
Download the datasets here and put them into data directory:
* Kvasir-SEG:
[Kvasir-SEG](https://www.kaggle.com/datasets/debeshjha1/kvasirseg)
* CVC-ClinicDB:
[CVC-ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)


1.) make directory named "dataset/'datasetname'"

2.) make three sub-directories "train" "val" "test"

3.) Put images under directory named "images"

4.) Put masks under directory named "masks"

Below is an example:
```
├─dataset
    ├─kvasir-seg
        ├─train
            ├─images
            ├─masks
        ├─val
            ├─images
            ├─masks
        └─test
            ├─images
            ├─masks
```
## 3. Download the models (loading models):

Download the pretrained 'PVTv2' model and put it into `lib/bkbone` directory.

* [models](https://pan.baidu.com/s/1piOOt5yYNLc_bVc6DPU2iA ) code: h25m

And the pretrained models of CGMA-Net on two ployp datasets are as follows:

* [models](https://pan.baidu.com/s/1rcXsAPTR67l9U5ZashbiFA) code: 48l2

please download and put them into checkpoints directory.

## 4. Train
```cmd
epoch=50 #default
lr=1e-4 #default
optimizer=AdamW #default
batchsize=8 #default
trainssize=352 #default
clip=0.5 #default
decay_rate=0.1 #default
train_path=dataset/{your_dataset_name}/train
test_path=dataset/{your_dataset_name}
train_save=checkpoint/{your_dataset_name}
Below is an example:
python lib/train.py --train_path dataset/kvasir-seg/train --test_path dataset/kvasir-seg --train_save checkpoint/kvasir-seg
```
    
    
## 5. Evaluate
```cmd
test_path=dataset/{your_dataset_name}/train
model_path=dataset/{your_dataset_name}/{your_model_name}
Below is an example:
python lib/train.py --test_path dataset/kvasir-seg/test --model_path checkpoint/kvasir-seg/CGMA.pth
```
    

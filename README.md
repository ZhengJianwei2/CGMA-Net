# CGMA-Net

## Papers
* CGMA-Net: CGMA-Net: Cross-Level Guidance and Multi-scale Aggregation Network for Polyp Segmentation

## 1. Download the datesets:
Download the datasets here:
* Kvasir-SEG:
[Kvasir-SEG](https://www.kaggle.com/datasets/debeshjha1/kvasirseg)
* CVC-ClinicDB:
[CVC-ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)

and put them into data directory.

1.) make directory named "dataset/'datasetname'"

2.) make three sub-directories "train" "val" "test"

3.) Put images under directory named "images"

4.) Put masks under directory named "masks"

## 2. Download the pretrained models:

* [models](https://pan.baidu.com/s/1rcXsAPTR67l9U5ZashbiFA) code: 48l2

and put them into checkpoints directory.

## 3. Train

    python lib/train.py
    
## 4. Test

    python lib/test.py



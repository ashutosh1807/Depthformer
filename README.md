# DEPTHFORMER: MULTISCALE VISION TRANSFORMER FOR MONOCULAR DEPTH ESTIMATION WITH GLOBAL LOCAL INFORMATION FUSION
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/depthformer-multiscale-vision-transformer-for/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=depthformer-multiscale-vision-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/depthformer-multiscale-vision-transformer-for/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=depthformer-multiscale-vision-transformer-for)

This is the official PyTorch implementation for ICIP 2022 paper 'Depthformer : Multiscale Vision Transformer For Monocular Depth Estimation With Local Global Information Fusion'.

## Abstract
Attention-based models such as transformers have shown outstanding performance on dense prediction tasks, such as semantic segmentation, owing to their capability of capturing long-range dependency in an image. However, the benefit of transformers for monocular depth prediction has seldom been explored so far. This paper benchmarks various transformer-based models for the depth estimation task on an indoor NYUV2 dataset and an outdoor KITTI dataset. We propose a novel attention-based architecture, Depthformer for monocular depth estimation that uses multi-head self-attention to produce the multiscale feature maps, which are effectively combined by our proposed decoder network. We also propose a Transbins module that divides the depth range into bins whose center value is estimated adaptively per image. The final depth estimated is a linear combination of bin centers for each pixel. Transbins module takes advantage of the global receptive field using the transformer module in the encoding stage. Experimental results on NYUV2 and KITTI depth estimation benchmark demonstrate that our proposed method improves the state-of-the-art by 3.3%, and 3.3% respectively in terms of Root Mean Squared Error (RMSE).

## Pretrained Models
* You can download the pretrained models "Depthformer_nyu.pt" and "Depthformer_kitti.pt" from [here](https://csciitd-my.sharepoint.com/:f:/g/personal/csy202452_iitd_ac_in/EkDava0AFO1LodDlo_fIAZEBKY4uPXTNLrCADh9na0z9jg?e=UBRQcv).

## Datset Preparation
We follow the dataset preparation strategy of [BTS](https://github.com/cleinc/bts).

## Training
NYUv2:
```
python train.py args_train_nyu_eigen.txt
```

KITTI:
```
python train.py args_train_kitti_eigen.txt
```

## Evaluation
NYUv2:
```
python evaluate.py args_test_nyu_eigen.txt
```

KITTI:
```
python evaluate.py args_test_kitti_eigen.txt
```
## Citation

If you find our work useful in your research, please cite the following:
```bibtex
@INPROCEEDINGS{9897187,
  author={Agarwal, Ashutosh and Arora, Chetan},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)}, 
  title={Depthformer: Multiscale Vision Transformer for Monocular Depth Estimation with Global Local Information Fusion}, 
  year={2022},
  volume={},
  number={},
  pages={3873-3877},
  doi={10.1109/ICIP46576.2022.9897187}}
```

## Contact
For questions about our paper or code, please contact ([@ashutosh1807](https://github.com/ashutosh1807)) or raise an issue on GitHub.

## References:
The code is adapted from the following repositories:

[1] <a href="https://github.com/shariqfarooq123/AdaBins.git">Adabins</a>
[2] <a href="https://github.com/NVlabs/SegFormer.git">Segformer</a>

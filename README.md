# domain-crossKD: Distilling GANs and Teacher Model for an Enhanced Semi-Supervised Object Detection

![Python 3.7](https://img.shields.io/badge/python-3.7-g)

This repository contains the official implementation of the following paper:
> **Distilling GANs and Teacher Model for an Enhanced Semi-Supervised Object Detection**<br>
> [Peter Chondro](https://scholar.google.co.uk/citations?hl=en&user=S9ErhhEAAAAJ)<sup>\*</sup> and [Jun-Ming Lu]  <br>
> *Independent, Industrial Technology Research Institute <br>

[[TechRxiv Paper](https://doi.org/10.36227/techrxiv.172954097.77246383/v1)]

## Introduction

Producing large, fully annotated datasets for object detection is highly expensive. In this paper, we propose a framework that integrates GANs and Knowledge Distillation to efficiently train object detectors using minimally annotated data. With a fully annotated source domain, 20% annotated target domain, and a large non-annotated target domain, the method employs GANs to generate synthetic target domain images and uses a teacher model to create pseudo-labels. Training the object detector with both detection and distillation losses resulted in a mean average precision (mAP) of 60.2% on the target domain, outperforming the 20% annotation baseline (51.8%) and closely matching the fully annotated baseline (62.8%).

![struture](assets/flow_diagram.png)

## Get Started

### 1. Prerequisites

**Dependencies**

- Ubuntu >= 16.04
- CUDA >= 11.3
- cuDNN (correspond to CUDA version)
- pytorch==2.4.0
- torchvision=0.19.0
- CMake >= 3.18
- OpenCV >= 4.0

### 2. Installation
**Step 0.** Clone this Repository
```shell
git clone https://github.com/peterchondro/domain-crossKD.git
cd domain-crossKD
```
**Step 1.** Create Conda Environment
```shell
conda create --name domain_crossKD python=3.7 -y
conda activate domain_crossKD
```
**Step 2.** Install Dependencies 
```shell
pip install -r requirements.txt --upgrade
```
**Step 3.** Training or Testing Domain Transfer
```shell
python3 domain_transfer/scripts/train.py
    --D_netDs projected_d basic\
    --D_proj_interp 532\
    --D_proj_network_type dinov2_vitb14\
    --G_nblocks 9\
    --G_netG mobile_resnet_attn\
    --G_ngf 128\
    --G_padding_type reflect\
    --alg_cut_nce_idt\
    --checkpoints_dir checkpoints/day2night\
    --data_crop_size 256\
    --data_dataset_mode unaligned_labeled_mask_online\
    --data_load_size 256\
    --data_online_creation_crop_delta_A 64\
    --data_online_creation_crop_delta_B 64\
    --data_online_creation_crop_size_A 256\
    --data_online_creation_crop_size_B 256\
    --data_relative_paths\
    --dataaug_no_rotate\
    --dataroot scripts/day2night/\
    --ddp_port 13458\
    --f_s_config_segformer models/configs/segformer/segformer_config_b1.json\
    --f_s_net segformer\
    --f_s_semantic_nclasses 19\
    --gpu 0\
    --model_input_nc 3\
    --model_output_nc 3\
    --model_type cut\
    --name bdd100k_day2night_256\
    --output_display_freq 100\
    --output_print_freq 100\
    --train_D_lr 0.0001\
    --train_G_ema\
    --train_G_lr 0.0002\
    --train_batch_size 1\
    --train_iter_size 8\
    --train_mask_f_s_B\
    --train_n_epochs 800\
    --train_optim adamw\
    --train_sem_idt\
    --train_sem_mask_lambda 10.0\
    --train_sem_use_label_B\
    --train_semantic_mask\
    --with_amp
python3 domain_transfer/scripts/test.py
```
**Step 4.** Training or Testing Knowledge Distillation
```shell
```

## Results
| **Method**         | Source   | Target   | All      |
|:------------------:|:--------:|:--------:|:--------:|
| **Lower Baseline** | 48.4%    | 55.2%    | 51.8%    |
| **Upper Baseline** | 57.2%    | 68.4%    | 62.8%    |
| **FitNets**        | 49.1%    | 57.6%    | 53.7%    |
| **FGD**            | 51.2%    | 56.8%    | 54.0%    |
| **LD**             | 53.7%    | 58.2%    | 55.9%    |
| **CrossKD**        | 54.9%    | 60.5%    | 57.7%    |
| **Proposed**       | 56.7%    | 63.7%    | 60.2%    |

![struture](assets/demo_bdd.png)
Sample Results for Berkeley Drive Dataset

![struture](assets/demo_idd.png)
Sample Results for India Drive Dataset

## Citation

If you find our repo useful for your research, please cite us:

```
@misc{chondro2024,
      title={Distilling GANs and Teacher Model for an Enhanced Semi-Supervised Object Detection}, 
      author={P. Chondro and J.-M. Lu},
      year={2024},
      eprint={pending},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact

For technical questions, please raise a topic in the Issues section.

## Acknowledgement

This repo is modified from the following codebase 
[JoliGEN](https://github.com/jolibrain/joliGEN).
[DarkNet](https://github.com/AlexeyAB/darknet).

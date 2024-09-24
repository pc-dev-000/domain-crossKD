# domain-crossKD: Distilling GANs and Teacher Model for an Enhanced Semi-Supervised Object Detection

![Python 3.7](https://img.shields.io/badge/python-3.7-g)

This repository contains the official implementation of the following paper:
> **Distilling GANs and Teacher Model for an Enhanced Semi-Supervised Object Detection**<br>
> [Peter Chondro](https://scholar.google.co.uk/citations?hl=en&user=S9ErhhEAAAAJ)<sup>\*</sup> and [Jun-Ming Lu]  <br>
> *Independent, Industrial Technology Research Institute <br>

[[Arxiv Paper](https://www.overleaf.com/project/6699dbef4e03d9be5ba1553f)]

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
**Step 3.** Training or Testing 
```shell
python3 domain_transfer/scripts/train.py
python3 domain_transfer/scripts/test.py
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

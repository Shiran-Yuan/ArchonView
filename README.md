# ArchonView

_Faster, more accurate, and scalable generative NVS, without pretrained 2D generative models!_

__Next-Scale Autoregressive Models are Zero-Shot Single-Image Object View Synthesizers__

by Shiran Yuan and Hao Zhao

[arXiv:2503.13588](https://arxiv.org/abs/2503.13588)

![AIR](https://github.com/user-attachments/assets/f9655065-f271-45a6-8e56-23c9c27f8763)

This is the official codebase for ArchonView, as in the paper "Next-Scale Autoregressive Models are Zero-Shot Single-Image Object View Synthesizers." Checkpoints, training data, and evaluation data are hosted [here](https://huggingface.co/datasets/anon8567671/ArchonView/). 

## Introduction

__ArchonView__ is the first generative novel view synthesis (NVS) model to be based on an autoregressive modeling paradigm. Why autoregression? Well, previous works (such as [Zero 1-to-3](https://github.com/cvlab-columbia/zero123) and [EscherNet](https://github.com/kxhit/EscherNet)) on generative NVS commonly fine-tune on Stable Diffusion to exploit 2D priors. However, this makes it difficult for them to scale up in size without a correspondingly scaled-up text-to-image checkpoint, which requires ginormous amouns of data and compute. In addition, diffusion requires multiple denoising steps, thus causing relatively long inference times (which is very problematic for centralized services and APIs). 

We propose using next-scale autoregression as a backbone instead, and discover that this way __no 2D pretraining is needed__! (contrary to the current consensus in generative NVS that fine-tuning based on models trained on 2D datasets is necessary) Using only 3D data from Objaverse (the same as the "fine-tuning" step of previous works), we are able to train a model from scratch that 1. __readily scales__ both data-wise and model-size-wise; 2. gives __significantly superior NVS results__ consistently across various standard benchmarks, including from difficult camera poses;and 3. is __several times faster__ than diffusion-based models! We hope this work could drive a paradigm shift towards generative backbones that are more scalable, efficient, and effective.

We believe in open source, and hereby provide all the materials one would need to reproduce our paper from scratch. This includes all used checkpoints, the training/validation/evaluation sets (already rendered and with pre-computed CLIP embeddings for your convenience), and the training/inference code. __We encourage you to try it out yourself__! Comments are very welcome!

![teaser](https://github.com/user-attachments/assets/64e8c0dc-672b-4df5-b641-b579ff4dc973)

![qualitative](https://github.com/user-attachments/assets/e24f1bda-4540-4427-b0ac-27692f6055ac)

## Reproduction

To set up the reproduction environment, run the following:
```bash
conda create -y -n archonview python=3.10
conda activate archonview
pip install -r requirements.txt
```

To run inference, use `demo.py` by changing the model depth, checkpoint path, and file paths. To run training, run the distributed training command below on your nodes. Each node must have access to the training code and data.
```bash
torchrun --nproc_per_node=[?] --nnodes=[?] --node_rank=[?] --master_addr=[?] --master_port=[?] train.py \
--depth=[?] --bs=[?] --ep=[?] --tblr=8e-5 --fp16=1 --alng=1e-5 --wpe=0.01 --twde=0.08  --data_path=[?]
```

## Data Convention

The training data (which can be downloaded at the provided link, but you can make your own by following the directions here) is split into training and validation sets. The data directory (whose path should be filled as the `data_path` argument in the training command) should contain two directories `train` and `val`, while each of those folders contain subdirectories which each correspond to a single object. Every object in the training set is represented by `{000..011}.png` (rendering), `{000..011}.npy` (camera pose), and `{000..011}e.npy` (precomputed CLIP embedding). 

The rendering files are 256x256 white-background renderings of the objects. We computed CLIP embeddings of those files and saved them as 768-dimensional numpy vectors in the corresponding embedding files. The camera poses are represented under the world-to-camera convention as 3x4 matrices. 

## Acknowledgements

Our codebase borrows from [VAR](https://github.com/FoundationVision/VAR). We would like to thank the authors of that work.

# ArchonView

![AIR](https://github.com/user-attachments/assets/f9655065-f271-45a6-8e56-23c9c27f8763)

This is the official codebase for ArchonView, as in the paper "Next-Scale Autoregressive Models are Zero-Shot Single-Image Object View Synthesizers." Checkpoints, training data, and evaluation data are hosted [here](https://huggingface.co/datasets/anon8567671/ArchonView/). 

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

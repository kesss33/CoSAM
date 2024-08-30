# Co-SAM: Continual Learning for Segment Anything Model Adaptation

CoSAM (Continual Segmentation Adaptation Model) is a deep learning framework designed to enhance segmentation models by incorporating continual learning techniques to adapt dynamically to new, unseen domains without forgetting previously learned information.

## Project Background
![Co-SAM Benchmark](figures/architecture4.png)
Traditional segmentation models often struggle with domain shifts that render them less effective outside their training environments. CoSAM addresses this limitation by integrating state-of-the-art continual learning strategies, enabling the model to maintain high performance across a range of challenging segmentation scenarios.

## Installation

To set up a local development environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/CoSAM.git


# Navigate to the project directory
cd CoSAM

# Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```


## Preparation

### Download Datasets
https://drive.google.com/file/d/1YC0u1LNrq26167XQILXisQpj1-vA1jQ6/view?usp=drive_link
![Datasets](figures/datasets_demo.png)

### Download Pre-Trained Checkpoints
Pre-trained SAM
https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view?usp=drive_link

Pre-trained HQ-SAM Decoder
https://drive.google.com/file/d/1cwieLjTZZCYcTdzYvOKq2UC__e_B9QN9/view?usp=drive_link

## Train
![Learning to Adapt](figures/architecture4.png)

```bash
# Naive (sequential adaptation)
python -m torch.distributed.launch --nproc_per_node=1 train_CL.py --CLmethod naive
# Joint-Training
python -m torch.distributed.launch --nproc_per_node=1 train_CL.py
# Lwf
python -m torch.distributed.launch --nproc_per_node=1 train_CL.py --CLmethod lwf --distill_weight 3
# ER
python -m torch.distributed.launch --nproc_per_node=1 train_CL.py --CLmethod er
# EWC
python -m torch.distributed.launch --nproc_per_node=1 train_CL.py --CLmethod ewc --ewc_weight 10
# L2P
python -m torch.distributed.launch --nproc_per_node=1 train_CL.py --CLmethod l2p
# L2A (ours)
python -m torch.distributed.launch --nproc_per_node=1 train_adapter_pool.py --buffer_size 10

```
## Evaluate
```bash
# L2P
python -m torch.distributed.launch --nproc_per_node=1 train_CL.py --eval --CLmethod l2p --restore-model "saved_ckpt"
# L2A (ours)
python -m torch.distributed.launch --nproc_per_node=1 train_adapter_pool.py --eval
# Others
python -m torch.distributed.launch --nproc_per_node=1 train_CL.py --eval --restore-model "saved_ckpt"
```

## Results


![Comparison](figures/comparison.png)

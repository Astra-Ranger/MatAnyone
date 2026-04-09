#!/bin/bash
#SBATCH --job-name=matanyone_train
#SBATCH --partition=NV_RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/matanyone_%j.out
#SBATCH --error=logs/matanyone_%j.err

cd /public/home/cit_siruiliang/project/MatAnyone

source ~/.bashrc
conda activate matanyone

# 第一次用这个环境时执行一次即可
# pip install -e .
# 再额外安装与你 CUDA 匹配的 torch / torchvision

mkdir -p slurm_logs

GPU=${SLURM_GPUS_ON_NODE:-8}
MASTER_PORT=25358

OMP_NUM_THREADS=${GPU} \
torchrun --standalone \
  --nnodes=1 \
  --nproc_per_node=${GPU} \
  --master_port=${MASTER_PORT} \
  matanyone/train.py \
  exp_id=matanyone_slurm

#!/bin/bash
#SBATCH --job-name=matanyone_train
#SBATCH --partition=NV_RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/matanyone_%j.out
#SBATCH --error=slurm_logs/matanyone_%j.err

set -euo pipefail

ROOT_DIR=/public/home/cit_siruiliang/project/MatAnyone
LOG_DIR="${ROOT_DIR}/slurm_logs"

infer_gpu_count() {
  local value
  local -a devices

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${#devices[@]}"
    return
  fi

  for value in "${SLURM_STEP_GPUS:-}" "${SLURM_JOB_GPUS:-}"; do
    if [[ -n "${value}" ]]; then
      IFS=',' read -r -a devices <<< "${value}"
      echo "${#devices[@]}"
      return
    fi
  done

  if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
    echo "${SLURM_GPUS_ON_NODE}"
    return
  fi

  echo "Unable to determine how many GPUs Slurm assigned to this job." >&2
  echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'" >&2
  echo "SLURM_STEP_GPUS='${SLURM_STEP_GPUS:-}'" >&2
  echo "SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}'" >&2
  echo "SLURM_GPUS_ON_NODE='${SLURM_GPUS_ON_NODE:-}'" >&2
  exit 1
}

cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

. /public/home/cit_siruiliang/miniconda3/etc/profile.d/conda.sh
conda activate matanyone

# 第一次用这个环境时执行一次即可
# pip install -e .
# 再额外安装与你 CUDA 匹配的 torch / torchvision

GPU_COUNT="$(infer_gpu_count)"
CPUS_PER_GPU=1
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" -ge "${GPU_COUNT}" ]]; then
  CPUS_PER_GPU=$(( SLURM_CPUS_PER_TASK / GPU_COUNT ))
  if [[ "${CPUS_PER_GPU}" -lt 1 ]]; then
    CPUS_PER_GPU=1
  fi
fi

MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "Job ${SLURM_JOB_ID:-unknown} running on ${SLURMD_NODENAME:-$(hostname)}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_COUNT=${GPU_COUNT}, OMP_NUM_THREADS=${CPUS_PER_GPU}, MASTER_PORT=${MASTER_PORT}"

OMP_NUM_THREADS="${CPUS_PER_GPU}" \
torchrun --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPU_COUNT}" \
  --master_port="${MASTER_PORT}" \
  matanyone/train.py \
  exp_id=matanyone_slurm

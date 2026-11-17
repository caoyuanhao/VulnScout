#!/bin/bash
#SBATCH --job-name=vs-lora
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.log
# LoRA fine-tuning of Qwen3-4B-Instruct-2507 on data/sft. Submit from the repository root.
CW=/courses/INFO7375.202710/students/$USER
module purge
module load explorer anaconda3/2024.06
source $CW/envs/vulnscout/bin/activate
export HF_HOME=$CW/hf_cache
nvidia-smi --query-gpu=name,memory.total --format=csv
python finetune/train_lora.py --model Qwen/Qwen3-4B-Instruct-2507 --data data/sft --out outputs/lora-qwen3-4b "$@"

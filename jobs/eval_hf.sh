#!/bin/bash
#SBATCH --job-name=vs-eval
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_%j.log
# Evaluates an open-weight backend on the held-out test questions with the frozen data cache.
#   sbatch jobs/eval_hf.sh hf:Qwen/Qwen3-4B-Instruct-2507
#   sbatch jobs/eval_hf.sh hf:Qwen/Qwen3-4B-Instruct-2507+outputs/lora-qwen3-4b
CW=/courses/INFO7375.202710/students/$USER
module purge
module load explorer anaconda3/2024.06
source $CW/envs/vulnscout/bin/activate
export HF_HOME=$CW/hf_cache VULNSCOUT_FROZEN=1 VULNSCOUT_AS_OF=$(cat data/AS_OF)
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m vulnscout.evaluate --backend "$1" "${@:2}"

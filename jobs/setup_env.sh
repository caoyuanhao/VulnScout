#!/bin/bash
#SBATCH --job-name=vs-setup
#SBATCH --partition=courses
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_%j.log
# Creates the training/inference environment. torch is pinned to 2.6.0 built for CUDA 11.8:
# the cluster driver supports CUDA <= 12.3, and transformers needs torch >= 2.6 for .bin checkpoints.
# Submit from the repository root:  sbatch jobs/setup_env.sh
set -e
CW=/courses/INFO7375.202710/students/$USER
module purge
module load explorer anaconda3/2024.06
export PIP_CACHE_DIR=$CW/pip_cache
rm -rf $CW/envs/vulnscout
python -m venv $CW/envs/vulnscout
source $CW/envs/vulnscout/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
echo "torch==2.6.0" > /tmp/torch-constraint-$USER.txt
pip install -c /tmp/torch-constraint-$USER.txt -r requirements.txt -r finetune/requirements.txt
python -c "import torch, transformers, peft; print('TORCH', torch.__version__, torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__)"
pytest -q

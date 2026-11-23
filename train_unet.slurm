#!/bin/bash -l
#SBATCH --job-name=DL_train_unet
#SBATCH --output=logs/DL_train_unet%j.out
#SBATCH --error=logs/DL_train_unet%j.err
#SBATCH --clusters=wice
#SBATCH --partition=gpu_v100
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

source /data/leuven/393/vsc39388/venvs/rd/bin/activate

module load FFmpeg/4.3.2-GCCcore-10.3.0

echo "Python:"
python --version

echo "Torch:"
python -c "import torch; print(torch.__version__)"

echo "CUDA:"
python -c "import torch; print(torch.cuda.is_available())"

echo "TorchCodec:"
python -c "import torchcodec; print(torchcodec.__version__)"

python train_unet_script.py
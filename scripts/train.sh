#!/bin/bash
#SBATCH --clusters=tinygpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1

cd $HOME/siconvnet
module load python/3.8-anaconda
source .venv/bin/activate
export PYTHONPATH=.
srun python3 scripts/train.py --model $1 --evaluation $2 --kernel-size $3 --interpolation $4 --lr $5 --seed $6
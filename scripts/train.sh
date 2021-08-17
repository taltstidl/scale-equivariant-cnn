#!/bin/bash
#SBATCH --clusters=tinygpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --export=NONE

cd $HOME/siconvnet
module load python/3.8-anaconda
source .venv/bin/activate
export PYTHONPATH=.
srun python3 scripts/train.py --model $1 --evaluation $2 --kernel-size $3 --interpolation $4 --learning-rate $5 --seed $6
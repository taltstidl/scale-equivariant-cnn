#!/bin/bash
#SBATCH --clusters=tinygpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --nice=100

cd $HOME/siconvnet
module load python/3.8-anaconda
source .venv/bin/activate
export PYTHONPATH=.
export MLFLOW_EXPERIMENT_NAME=hyperparam
export MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow\$2142@mad-vm-thomas.informatik.uni-erlangen.de/mlflow_siconv
for seed in {1..50}
do
  srun python3 scripts/train.py --model $1 --evaluation $2 --kernel-size $3 --interpolation $4 --lr $5 --seed $seed
done
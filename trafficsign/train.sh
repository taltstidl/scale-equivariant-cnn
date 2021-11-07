#!/bin/bash
#SBATCH --clusters=tinygpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

cd $HOME/siconvnet
module load python/3.8-anaconda
source .venv/bin/activate
export PYTHONPATH=.
export MLFLOW_EXPERIMENT_NAME=trafficsign
export MLFLOW_TRACKING_URI=http://mad-vm-thomas.informatik.uni-erlangen.de/mlflow_siconv
for seed in {1..10}
do
  srun python3 trafficsign/train.py --model $1 --evaluation $2 --lr 1e-2 --seed 42
done
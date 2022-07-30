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
export MLFLOW_EXPERIMENT_NAME=trial2
export MLFLOW_TRACKING_URI=http://mad-vm-thomas.informatik.uni-erlangen.de/mlflow_siconv
for seed in {1..$7}
do
  srun python3 scripts/train.py --model $1 --data $2 --evaluation $3 --kernel-size $4 --interpolation $5 --lr $6 --seed $seed
done
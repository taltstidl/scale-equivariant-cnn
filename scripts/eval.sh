#!/bin/bash
#SBATCH --clusters=tinygpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

cd $HOME/siconvnet
module load python/3.8-anaconda
source .venv/bin/activate
export PYTHONPATH=.

srun python3 scripts/eval.py --runs runs-mlflow.csv --models mlruns
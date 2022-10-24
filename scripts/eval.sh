#!/bin/bash
#SBATCH --clusters=tinygpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

cd $HOME/siconvnet
module load python/3.8-anaconda
source .venv/bin/activate
export PYTHONPATH=.

evals=( --generalization --equivariance --index-correlation )
srun python3 scripts/eval.py --runs runs-mlflow.csv --models mlruns --data emoji "${evals[@]}"
srun python3 scripts/eval.py --runs runs-mlflow.csv --models mlruns --data mnist "${evals[@]}"
srun python3 scripts/eval.py --runs runs-mlflow.csv --models mlruns --data trafficsign "${evals[@]}"
srun python3 scripts/eval.py --runs runs-mlflow.csv --models mlruns --data aerial "${evals[@]}"
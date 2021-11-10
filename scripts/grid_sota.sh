#!/bin/bash
# Convenience batch file for grid search, launches each in a separate job

models=( "ensemble" "spatial_transform" )
evaluations=( 1 2 3 4 5 )

for model in "${models[@]}"
do
  for evaluation in "${evaluations[@]}"
  do
    sbatch train_sota.sh $model $evaluation 7 bicubic 1e-2
  done
done
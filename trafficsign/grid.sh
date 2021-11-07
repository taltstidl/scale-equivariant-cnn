#!/bin/bash
# Convenience batch file for grid search, launches each in a separate job

models=( "standard" "scale_equiv" "spatial_transformer" "ensemble" )
evaluations=( 1 2 3 )

for model in "${models[@]}"
do
  for evaluation in "${evaluations[@]}"
  do
    sbatch train.sh $model $evaluation
  done
done
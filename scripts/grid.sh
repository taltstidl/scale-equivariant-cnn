#!/bin/bash
# Convenience batch file for grid search, launches each in a separate job

models=( "standard" "pixel_pool" "slice_pool" "conv3d" )
evaluations=( 1 2 3 4 5 )
interpolations=( "nearest" "bilinear" "bicubic" "area" )
seeds=( 42 88 38 52 26 )

for model in "${models[@]}"
do
  for evaluation in "${evaluations[@]}"
  do
    for interpolation in "${interpolations[@]}"
    do
      for seed in "${seeds[@]}"
      do
        sbatch train.sh $model $evaluation $interpolation $seed
      done
    done
  done
done
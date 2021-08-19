#!/bin/bash
# Convenience batch file for grid search, launches each in a separate job

models=( "standard" "pixel_pool" "slice_pool" "conv3d" )
evaluations=( 1 2 3 4 5 )
kernel_sizes=( 3 7 11 15 )
interpolations=( "nearest" "bilinear" "bicubic" "area" )

for model in "${models[@]}"
do
  for evaluation in "${evaluations[@]}"
  do
    for kernel_size in "${kernel_sizes[@]}"
    do
      for interpolation in "${interpolations[@]}"
      do
        sbatch train.sh $model $evaluation $kernel_size $interpolation 1e-2
      done
    done
  done
done
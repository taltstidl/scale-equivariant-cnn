#!/bin/bash
# Convenience batch file for grid search, launches each in a separate job

models=( "standard" "pixel_pool" "slice_pool" "conv3d" )
evaluations=( 1 2 3 4 5 )
kernel_sizes = ( 3 7 11 15 )
interpolations=( "nearest" "bilinear" "bicubic" "area" )
learning_rates = { 1..50 }
seeds=( 42 88 38 52 26 )

for model in "${models[@]}"
do
  for evaluation in "${evaluations[@]}"
  do
    for kernel_size in "${kernel_sizes[@]}"
    do
      for interpolation in "${interpolations[@]}"
      do
        for learning_rate in "${learning_rates[@]}"
        do
          for seed in "${seeds[@]}"
          do
            sbatch train.sh $model $evaluation $kernel_size $interpolation $learning_rate $seed
          done
        done
      done
    done
  done
done
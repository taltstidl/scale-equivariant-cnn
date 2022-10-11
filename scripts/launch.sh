#!/bin/bash
# Convenience batch file for experiments, launches each in a separate job

models=( "standard" "pixel_pool" "slice_pool" "conv3d" "ensemble" "spatial_transform" )
datasets=( "emoji" "mnist" "trafficsign" )
evaluations=( 1 2 3 )

for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    for evaluation in "${evaluations[@]}"
    do
      sbatch train.sh $model $dataset $evaluation 7 bicubic 1e-2 1 50
    done
  done
done
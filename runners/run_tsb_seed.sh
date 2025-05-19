#!/bin/bash

for g in $(seq 225 236)
do
  for seed in $(seq 47 50)
  do
    python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 128 --random_seed $seed
  done
done


for g in $(seq 237 256)
do
  for seed in $(seq 47 50)
  do
    python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 128 --random_seed $seed
  done
done

for g in $(seq 260 276)
do
  for seed in $(seq 47 50)
  do
    python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 128 --random_seed $seed
  done
done

for g in $(seq 287 301)
do
  for seed in $(seq 47 50)
  do
    python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 128 --random_seed $seed
  done
done
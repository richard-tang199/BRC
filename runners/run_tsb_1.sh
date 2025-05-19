#!/bin/bash

for g in $(seq 225 236)
do
  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
done

for g in $(seq 237 256)
do
  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
done

for g in $(seq 260 276)
do
  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
done

for g in $(seq 287 301)
do
  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
done

#!/bin/bash

for g in $(seq 225 236)
do
#  python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_hard --device "cuda:3" --plot False --num_epochs 300  --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_gating --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name time_soft --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_conv --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_add  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_point --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4 &
  python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_window --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 1 &
  python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name guide_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4
done

for g in $(seq 237 256)
do
#  python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_hard --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_gating --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name time_soft --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#   python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#   python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_conv --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_add  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#   python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_point --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4 &
   python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_window --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 1 &
   python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name guide_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4
done

for g in $(seq 260 276)
do
#  python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_hard --device "cuda:3" --plot False --num_epochs 300  --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_gating --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name time_soft --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#   python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#   python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_conv --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_add  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#   python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_point --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4 &
   python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_window --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 1 &
   python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name guide_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4
done

for g in $(seq 287 301)
do
#  python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_hard --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_gating --device "cuda:3" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name time_soft --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_tf_conv --device "cuda:3" --plot False --num_epochs 300 --batch_size 128
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_add  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64
#   python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_point --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4
   python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_window --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 1 &
   python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name guide_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4
done

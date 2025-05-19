for g in $(seq 1 251)
do
  for omega in 0.1 1 5 10 100
  do
    python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_tf_v2  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --omega $omega
  done

  for patch_size in 4 6 8 10 12
  do
    python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_tf_v2  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --patch_size $patch_size
  done

  for window_multiple in 1 2 4 6 8
  do
    python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_tf_v2  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple $window_multiple
  done
done


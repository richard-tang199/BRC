

for g in $(seq 12 250)
do
  for seed in $(seq 47 50)
  do
    python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_tf_v2 --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --random_seed $seed
#    python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
#    python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
#    python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_time_add --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
  #   python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_tf_conv --device "cuda:0" --plot True --num_epochs 300 --batch_size 64
  #  python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_tf_v2 --device "cuda:3" --plot True --num_epochs 300 --batch_size 64
  #  python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_hard --device "cuda:0" --plot True --num_epochs 300 --batch_size 128
  #  python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_gating --device "cuda:0" --plot True --num_epochs 300 --batch_size 128
  #  python train_classify_mask.py --dataset_name UCR --group_name $g --model_name time_soft --device "cuda:0" --plot True --num_epochs 300 --batch_size 128
  done
done

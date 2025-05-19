for g in $(seq 1 12)
#for g in "${group_list[@]}"
do
  python train_classify_mask.py --dataset_name ASD --group_name $g --model_name freq_new --device "cuda:0" --num_epochs 100 --batch_size 16
  python train_classify_mask.py --dataset_name ASD --group_name $g --model_name freq_hard --device "cuda:0" --num_epochs 100 --batch_size 16
  python train_classify_mask.py --dataset_name ASD --group_name $g --model_name freq_gating --device "cuda:0" --plot True --num_epochs 100 --batch_size 16
  python train_classify_mask.py --dataset_name ASD --group_name $g --model_name time_soft --device "cuda:0" --plot True --num_epochs 100 --batch_size 16
done
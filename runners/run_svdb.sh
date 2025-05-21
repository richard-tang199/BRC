for g in $(seq 237 256)
#for g in "${group_list[@]}"
do
  python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_new --device "cuda:0"
  python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_hard --device "cuda:0"
done
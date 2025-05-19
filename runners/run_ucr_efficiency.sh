for g in $(seq 1 250)
do
#  python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_new --device "cuda:3" --num_epochs 300 --batch_size 64
#  python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_tf --device "cuda:0" --plot True --num_epochs 300 --batch_size 64
#   python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
#   python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_gating --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
#  python train_classify_mask.py --dataset_name UCR --group_name $g --model_name time_soft --device "cuda:0" --plot True --num_epochs 300 --batch_size 128
   python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_tf_v2 --device "cuda:0" --plot False --num_epochs 0 --batch_size 64
done
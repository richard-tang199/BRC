# alation study for CoAD
# freq_time_mean: the classification result is the mean of the frequency and time branches
# freq_time_gate: the classification result is the gate fusions of the frequency and time features
# freq_time_add: the classification result is the addition of the frequency and time features
# freq_time_point: the classification result is the step-wise classification granularity
# freq_time_window: the classification result is the window-wise classification granularity
# guide_hard: the reconstruction module is guided by the hard masking strategy
# freq_hard: the reconstruction module is used with random masking strategy without guiding
# freq_gating: the reconstruction module is used with gating strategy without guiding
# time_soft: the classification module only takes the time feature into account without frequency feature

for g in $(seq 1 250)
do
 python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
 pyhton train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
 python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_time_add --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
 python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_time_point --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4
 python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_time_window --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 1
 python train_classify_mask.py --dataset_name UCR --group_name $g --model_name guide_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 128 --window_multiple 4
 python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
 python train_classify_mask.py --dataset_name UCR --group_name $g --model_name freq_gating --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
 python train_classify_mask.py --dataset_name UCR --group_name $g --model_name time_soft --device "cuda:0" --plot False --num_epochs 300 --batch_size 128
done

for g in $(seq 225 236)
do
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_hard --device "cuda:3" --plot False --num_epochs 300  --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_gating --device "cuda:3" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name time_soft --device "cuda:3" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_add  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_point --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 4 & 
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_window --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 1 &
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name guide_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 4
 wait
done

for g in $(seq 237 256)
do
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_hard --device "cuda:3" --plot False --num_epochs 300  --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_gating --device "cuda:3" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name time_soft --device "cuda:3" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_add  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_point --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 4 & 
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_window --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 1 &
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name guide_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 4
 wait
done

for g in $(seq 260 276)
do
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_hard --device "cuda:3" --plot False --num_epochs 300  --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_gating --device "cuda:3" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name time_soft --device "cuda:3" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_add  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_point --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 4 & 
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_window --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 1 &
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name guide_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 4
 wait
done

for g in $(seq 287 301)
do
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_hard --device "cuda:3" --plot False --num_epochs 300  --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_gating --device "cuda:3" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name time_soft --device "cuda:3" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_mean --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_gate --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U  --group_name $g --model_name freq_time_add  --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 &
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_point --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 4 & 
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name freq_time_window --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 1 &
 python train_classify_mask.py --dataset_name TSB-AD-U --group_name $g --model_name guide_hard --device "cuda:0" --plot False --num_epochs 300 --batch_size 64 --window_multiple 4
 wait
done
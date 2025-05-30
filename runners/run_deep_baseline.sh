for g in $(seq 1 250)
do
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name cutaddpaste --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name couta --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name fcvae --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name catch --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name tfad --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name dada --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name moment --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name TFMAE --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name m2n2 --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name TranAD --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name UCR --model_name MAUT --plot False --device cuda:0
done

for g in $(seq 225 236)
do
python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name cutaddpaste --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name couta --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name fcvae --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name catch --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name tfad --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name dada --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name moment --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name TFMAE --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name m2n2 --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name TranAD --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name MAUT --plot False --device cuda:0
done

for g in $(seq 237 256)
do
python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name cutaddpaste --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name couta --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name fcvae --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name catch --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name tfad --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name dada --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name moment --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name TFMAE --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name m2n2 --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name TranAD --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name MAUT --plot False --device cuda:0
done 

for g in $(seq 260 276)
do
python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name cutaddpaste --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name couta --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name fcvae --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name catch --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name tfad --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name dada --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name moment --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name TFMAE --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name m2n2 --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name TranAD --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name MAUT --plot False --device cuda:0
done 

for g in $(seq 287 301)
do
python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name cutaddpaste --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name couta --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name fcvae --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name catch --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name tfad --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name dada --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name moment --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name TFMAE --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name m2n2 --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name TranAD --plot False --device cuda:0
 python deep_baseline_compare.py --group $g --dataset_name TSB-AD --model_name MAUT --plot False --device cuda:0
done 
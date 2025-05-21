for g in $(seq 1 250)
do
  python baseline_compare.py --model_name damp --dataset_name UCR --group_name $g
done
for g in $(seq 1 250)
do
#  python baseline_compare.py --model_name KshapeAD --dataset_name UCR --group_name $g &
  python baseline_compare.py --model_name Series2Graph --dataset_name UCR --group_name $g &
  python baseline_compare.py --model_name Kmeans --dataset_name UCR --group_name $g &
  python baseline_compare.py --model_name MP --dataset_name UCR --group_name $g &
  python baseline_compare.py --model_name PCA --dataset_name UCR --group_name $g &
  python baseline_compare.py --model_name SAND --dataset_name UCR --group_name $g &
  python baseline_compare.py --model_name damp --dataset_name UCR --group_name $g
done
for g in $(seq 225 236)
do
#  python baseline_compare.py --model_name SAND --group_name $g
#  python baseline_compare.py --model_name Series2Graph --group_name $g
#  python baseline_compare.py --model_name Kmeans --group_name $g
#  python baseline_compare.py --model_name damp  --group_name $g
#  python baseline_compare.py --model_name MP --group_name $g
#  python baseline_compare.py --model_name PCA --group_name $g
   python baseline_compare.py --model_name SAND --dataset_name TSB-AD-U --group_name $g
done

for g in $(seq 237 256)
do
#  python baseline_compare.py --model_name SAND --group_name $g
#  python baseline_compare.py --model_name Series2Graph --group_name $g
#  python baseline_compare.py --model_name Kmeans --group_name $g
#  python baseline_compare.py --model_name damp  --group_name $g
#  python baseline_compare.py --model_name MP --group_name $g
#python baseline_compare.py --model_name PCA --group_name $g
   python baseline_compare.py --model_name SAND --dataset_name TSB-AD-U --group_name $g
done

for g in $(seq 260 276)
do
#  python baseline_compare.py --model_name SAND --group_name $g
#  python baseline_compare.py --model_name Series2Graph --group_name $g
#  python baseline_compare.py --model_name Kmeans --group_name $g
#  python baseline_compare.py --model_name damp  --group_name $g
#  python baseline_compare.py --model_name MP --group_name $g
#  python baseline_compare.py --model_name PCA --group_name $g
   python baseline_compare.py --model_name SAND --dataset_name TSB-AD-U --group_name $g
done

for g in $(seq 287 301)
do
#  python baseline_compare.py --model_name SAND --group_name $g
#  python baseline_compare.py --model_name Series2Graph --group_name $g
#  python baseline_compare.py --model_name Kmeans --group_name $g
#  python baseline_compare.py --model_name damp  --group_name $g
#  python baseline_compare.py --model_name MP --group_name $g
#python baseline_compare.py --model_name PCA --group_name $g
   python baseline_compare.py --model_name SAND --dataset_name TSB-AD-U --group_name $g
done




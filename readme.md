<h1 align="center">CoAD</h1>
<h2 align="center">🌉 Bridging Classification and Reconstruction: Cooperative Time Series Anomaly Detection</h2>

## Overall Framework ##
<p style="text-align: center;">
<img src="fig/main_structure.png" alt="main_structure" width="500">
</p>

## Main Results ##
Evaluation results on reliable datasets (KDD21 [1] and TSB-AD [2]) using rigorous evaluation protocols [2].

<p style="text-align: center;">
<img src="fig/main_results.png" alt="main_results" width="500">
</p>

## Case Studies ##
Visualizes the detection results of COAD on several challenging cases.

<p style="text-align: center;">
<img src="fig/case_study.png" alt="main_results" width="500">
</p>


## Setup ##
Installation
```
conda create -n MMA python=3.11
conda activate MMA
pip install -r requirements.txt
```

## Prepare datasets ##

Download the dataset from the anonymous link [dataset](https://d.kuku.lu/pfj2vscrj) and extract it to the `dadaset` folder.
```
├─dataset
├───TSB-AD

└───UCR
```

## 

## References ##
1. E. Keogh, “Multidataset time series anomaly detection competition,” 2021, https://compete.hexagon-ml.com/practice/competition/39/.

2. Q. Liu and J. Paparrizos, “The elephant in the room: Towards a reliable time-series anomaly detection
358 benchmark,” in The 38th Conference on Neural Information Processing Systems Datasets and Benchmarks
359 Track, 2024

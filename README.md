## [2024 ICME Grand Challenge: Multi-Modal Video Reasoning and Analyzing Competition (MMVRAC)](https://sutdcv.github.io/MMVRAC/)

dataset: [UAVhuman-reid](https://github.com/sutdcv/UAV-Human)

### 1. Configurations
First of all, create a conda env, then install packages detailed in enviroments.sh
```
conda create -n reid python==3.9
conda activate reid
bash enviroments.sh
```

Note that, all experiments are conducted using single GPU: NVIDIA Titan RTX.

### 2. training
Modify the paths and settings in config/uavhuman.yml, then

```
python train.py --config_file config/#your_config_name#.yml   
```

### 3. evaluation
Modify TEST.WEIGHT as your trained model path in config/uavhuman.yml, then

```
python test_attribute_recognition.py --config_file config/#your_config_name#.yml
```

Otherwise, specifying the trained path in terminal is also valid:

```
python test_attribute_recognition.py --config_file config/#your_config_name#.yml TEST.WEIGHT ${your trained model path}
```



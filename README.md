## [2024 ICME Grand Challenge: Multi-Modal Video Reasoning and Analyzing Competition (MMVRAC)](https://sutdcv.github.io/MMVRAC/)

dataset: [UAVhuman-reid](https://github.com/sutdcv/UAV-Human)

The ultimate results of our method is presented in **6.Results**.

### 1. Configurations
First of all, create a conda env, then install packages detailed in enviroments.sh
```
conda create -n reid python==3.9
conda activate reid
bash enviroments.sh
```

Note that, all experiments are conducted using single GPU: NVIDIA Titan RTX. All experiments are based on the vit_base model. 

### 2. Google Drive Link

Google Drive Link: https://drive.google.com/drive/folders/1leOMxBshwHRCndR5y15LI5VBKc1z8Xqt?usp=sharing

### 3. training

Modify the paths and settings in related yml file in config folder, then
#### 3.1 training Backpack

```
python train.py --config_file config/train_Backpack.yml
```
#### 3.2 training Hat

Modify function _process_dir in data/datasets/uavhuman.py begin at lines 240, change set_LCC_number to true, then

```
python train.py --config_file config/train_Hat.yml
```
#### 3.3 traing UCC

Modify function _process_dir in data/datasets/uavhuman.py begin at lines 240, change set_UCC_number to true, then

```
python train.py --config_file config/train_UCC.yml
```
#### 3.4 training UCS

Modify function _process_dir in data/datasets/uavhuman.py begin at lines 240, change set_UCC_number to true, then

```
python train.py --config_file config/train_UCS.yml
```
#### 3.5 traing LCC

Modify function _process_dir in data/datasets/uavhuman.py begin at lines 240, change set_LCC_number_1 to true, then

```
python train.py --config_file config/train_LCC.yml
```
#### 3.6 traing LCS

Modify function _process_dir in data/datasets/uavhuman.py begin at lines 240, change set_UCC_number to true, then

```
python train.py --config_file config/train_LCS.yml
```

#### 3.7 training Gender

Use LLAVA to identify the gender of the image.

LLaVA link:[haotian-liu/LLaVA: [NeurIPS'23 Oral\] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond. (github.com)](https://github.com/haotian-liu/LLaVA)

LLaVA model link: [liuhaotian/llava-v1.5-7b · Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-7b)

Prompt for Gender:

```
"Please use male or female to indicate the gender of the person in the picture."
```

Generate result json files in https://drive.google.com/drive/folders/1GVqlaJOORDNi1SZwxOjM9pMxb72nou5o?usp=sharing

### 4. evaluation

Trained models in https://drive.google.com/drive/folders/1ocP_bToUrLYh89EV7BQtsCIbcjcZNbLE?usp=sharing

Modify TEST.WEIGHT as your trained model path in config/test_attr.yml, then
#### 4.1 for UCS evaluation
First set MODEL.NAME = "attr_vit".
Second set MODEL.TRANSFORMER_TYPE = "attr_vit_base_patch16_224_TransReID".
Then,  
```
python test_attribute_recognition.py --config_file config/test_attr.yml
```
#### 4.2 for Gender evaluation

Replace file_path in convert_llava_json_to_label.py, then 

```python
python convert_llava_json_to_label.py
```

#### 4.3 for other attribut evaluation

First set MODEL.NAME = "only_attribute_recognition".
Second set MODEL.TRANSFORMER_TYPE = "only_attr_vit_base_patch16_224_TransReID".
Then,  
```
python test_attribute_recognition.py --config_file config/test_attr.yml
```

### 5. Re-recognition to Improve Results

We use this trick to improve the accuracy for backpack, hat, UCC and LCC attributes. Our starting point for using this method is that we believe that the attributes of the same person should be consistent. Therefore, we use the best model trained by Reid to predict the distance between positive examples(we believe that this sample has a high recognition accuracy rate, so it is set as a positive example) and negative examples(we believe that this sample has a high recognition error rate, so it is set as a negative example), so that the negative examples which less than the set threshold have the same attribute label as the corresponding positive examples. The following files are used in this process:

Json files in https://drive.google.com/drive/folders/1ocP_bToUrLYh89EV7BQtsCIbcjcZNbLE?usp=sharing

The result json file after re-retreival are in  https://drive.google.com/drive/folders/1ugporKPX1vc4rC1jddx3B7j6RlqmHA19?usp=sharing

First, modify TEST.WEIGHT as our reid trained model path in config/test.yml.

Second, modify file_path as our related json file in google link.

Then,  

#### 5.1 Re-recognition for Backpack

Modify "__init__" in data/datasets/uavhuman.py begin at lines 24, change backpack_attempt to true, then

```
python test.py --config_file config/test.yml
```

#### 5.2 Re-recognition for Hat

Modify "__init__" in data/datasets/uavhuman.py begin at lines 24, change hat_attempt to true, then

```
python test.py --config_file config/test.yml
```

#### 5.3 Re-recognition for UCC

##### 5.3.1 First Attempt

Modify "__init__" in data/datasets/uavhuman.py begin at lines 24, change UCC_first_attempt to true, then

```
python test.py --config_file config/test.yml
```

##### 5.3.2 Second Attempt

Json file https://drive.google.com/file/d/15kPsxr6FSY1G2l_GllxU6Dps3o4MKFYo/view?usp=sharing

Modify "__init__" in data/datasets/uavhuman.py begin at lines 24, change UCC_second_attempt to true, then

```
python test.py --config_file config/test.yml
```

#### 5.4 Re-recognition for LCC

Modify "__init__" in data/datasets/uavhuman.py begin at lines 24, change LCC_attempt to true, then

```
python test.py --config_file config/test.yml
```

### 6.Results

| **Attribute**   | **Accuracy** |
| --------------- | ------------ |
| **1. Gender**   | **94.75**    |
| **2. Backpack** | **84.33**    |
| **3. Hat**      | **83.46**    |
| **4. UCC**      | **72.07**    |
| **5. UCS**      | **93.93**    |
| **6. LCC**      | **90.68**    |
| **7. LCS**      | **92.51**    |
| **Total**       | **611.73**   |


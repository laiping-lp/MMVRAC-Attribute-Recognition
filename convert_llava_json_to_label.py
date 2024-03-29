import os
import json
from tqdm import tqdm
import re
from prettytable import PrettyTable

pattern_pid = re.compile(r'P([-\d]+)S([-\d]+)')
pattern_gender = re.compile(r'G([-\d]+)')
pattern_backpack = re.compile(r'B([-\d]+)')
pattern_hat = re.compile(r'H([-\d]+)')
pattern_upper = re.compile(r'UC([-\d]+)')
pattern_lower = re.compile(r'LC([-\d]+)')
pattern_action = re.compile(r'A([-\d]+)')

file_path_1 = "//data3/laiping/datasets/UAVHuman/llava-attr/Gender_Backpack_Hat_UAVHuman_test.json"
file_path_2 = "/data3/laiping/datasets/UAVHuman/llava-attr/UCC_UCS_UAVHuman_test.json"
file_path_3 = "/data3/laiping/datasets/UAVHuman/llava-attr/LCC_LCS_UAVHuman_test.json"

with open(file_path_1,'r') as f1:
    load_data_1 = json.load(f1)
with open(file_path_2,'r') as f2:
    load_data_2 = json.load(f2)
with open(file_path_3,'r') as f3:
    load_data_3 = json.load(f3)

counts = {
    "total_number":0,
    "total_gender_number":0,
    "gender":0,
    "total_backpack_number":0,
    "backpack":0, 
    "total_hat_number":0,
    "hat":0,
    "total_UCC_number":0,
    "upper_color":0, 
    "total_UCS_number":0,
    "upper_style":0,
    "total_LCC_number":0,
    "lower_color":0,
    "total_LCS_number":0,
    'lower_style':0
}
counts['total_number'] = len(load_data_1)

gender_number = [0] * 2
backpack_number = [0] * 5
hat_number = [0] * 5
UCC_number = [0] * 12
UCS_number = [0] * 4
LCC_number = [0] * 12
LCS_number = [0] * 4


for data in tqdm(load_data_1):
    query_filename = os.path.basename(data["file_path"])
    q_gender = int(pattern_gender.search(query_filename).groups()[0][0]) # 0: n/a; 1: male; 2: female
    q_backpack = int(pattern_backpack.search(query_filename).groups()[0][0])
    if q_backpack == 5:
        q_backpack = 0 # 0: n/a; 1: red; 2: black; 3: green; 4: yellow; 5: n/a
    q_hat = int(pattern_hat.search(query_filename).groups()[0][0])
    if q_hat == 5:
        q_hat = 0
    gender_data = data['pre_label_gender'].lower()
    if "female" in gender_data:
        pre_gender = 2
        gender_number[pre_gender - 1] += 1
    else:
        pre_gender = 1
        gender_number[pre_gender - 1] += 1
    if pre_gender == q_gender:
        counts['gender'] += 1
    backpack_data = data['pre_label_backpack'].lower()
    if 'red' in backpack_data:
        pre_backpack = 1
        backpack_number[pre_backpack] +=1
    elif 'black' in backpack_data:
        pre_backpack = 2
        backpack_number[pre_backpack] +=1
    elif 'green' in backpack_data:
        pre_backpack = 3
        backpack_number[pre_backpack] +=1
    elif 'yellow' in backpack_data:
        pre_backpack = 4
        backpack_number[pre_backpack] +=1
    elif 'n/a' or 'no' in backpack_data:
        pre_backpack = 0
        backpack_number[pre_backpack] +=1
    if pre_backpack == q_backpack:
        counts['backpack'] += 1 
    hat_data = data['pre_label_hat'].lower()
    if 'red' in hat_data:
        pre_hat = 1
        hat_number[pre_hat] +=1
    elif 'black' in hat_data:
        pre_hat = 2
        hat_number[pre_hat] +=1
    elif 'yellow' in hat_data:
        pre_hat = 3
        hat_number[pre_hat] +=1
    elif 'white' in hat_data:
        pre_hat = 4
        hat_number[pre_hat] +=1
    elif 'n/a' or 'no' in hat_data:
        pre_hat = 0
        hat_number[pre_hat] +=1
    if pre_hat == q_hat:
        counts['hat'] += 1 

       
for data in tqdm(load_data_2):
    query_filename = os.path.basename(data["file_path"])
    q_upper_cloth = pattern_upper.search(query_filename).groups()[0][:3]
    q_upper_color = int(q_upper_cloth[:2]) # 0: n/a; 1: red; 2: black; 3: blue; 4: green; 5: multicolor; 6: grey; 7: white; 8: yellow; 9: dark brown; 10: purple; 11: pink
    q_upper_style = int(q_upper_cloth[2]) # 0: n/a; 1: long; 2: short; 3: skirt
    UCC_data = data['pre_label_UCC'].lower()
    if 'red' in UCC_data:
        pre_UCC = 1
        UCC_number[pre_UCC] +=1
    elif 'black' in UCC_data:
        pre_UCC = 2
        UCC_number[pre_UCC] +=1
    elif 'blue' in UCC_data:
        pre_UCC = 3
        UCC_number[pre_UCC] +=1
    elif 'green' in UCC_data:
        pre_UCC = 4
        UCC_number[pre_UCC] +=1
    elif 'multicolor' in UCC_data:
        pre_UCC = 5
        UCC_number[pre_UCC] +=1
    elif 'grey' in UCC_data:
        pre_UCC = 6
        UCC_number[pre_UCC] +=1
    elif 'white' in UCC_data:
        pre_UCC = 1
        UCC_number[pre_UCC] +=1
    elif 'yellow' in UCC_data:
        pre_UCC = 8
        UCC_number[pre_UCC] +=1
    elif 'dark brown' in UCC_data:
        pre_UCC = 9
        UCC_number[pre_UCC] +=1
    elif 'purple' in UCC_data:
        pre_UCC = 10
        UCC_number[pre_UCC] +=1
    elif 'pink' in UCC_data:
        pre_UCC = 11
        UCC_number[pre_UCC] +=1
    elif 'n/a' or 'no' in UCC_data:
        pre_UCC = 0
        UCC_number[pre_UCC] +=1
    if pre_UCC == q_upper_color:
        counts['upper_color'] += 1
    UCS_data = data['pre_label_UCS'].lower()
    if 'long' in UCS_data:
        pre_UCS = 1
        UCS_number[pre_UCS] +=1
    elif 'short' in UCS_data:
        pre_UCS = 2
        UCS_number[pre_UCS] +=1
    elif 'skirt' in UCS_data:
        pre_UCS = 3
        UCS_number[pre_UCS] +=1
    elif 'n/a' or 'no' in UCS_data:
        pre_UCS = 4
        UCS_number[pre_UCS] +=1
    if pre_UCS == q_upper_style:
        counts['upper_style'] +=1 
for data in tqdm(load_data_3):
    query_filename = os.path.basename(data["file_path"])
    q_lower_cloth = pattern_lower.search(query_filename).groups()[0][:3]
    q_lower_color = int(q_lower_cloth[:2]) # 0: n/a; 1: red; 2: black; 3: blue; 4: green; 5: multicolor; 6: grey; 7: white; 8: yellow; 9: dark brown; 10: purple; 11: pink
    q_lower_style = int(q_lower_cloth[2]) # 0: n/a; 1: long; 2: short; 3: skirt
    LCC_data = data['pre_label_LCC'].lower()
    if 'red' in LCC_data:
        pre_LCC = 1
        LCC_number[pre_LCC] +=1
    elif 'black' in LCC_data:
        pre_LCC = 2
        LCC_number[pre_LCC] +=1
    elif 'blue' in LCC_data:
        pre_LCC = 3
        LCC_number[pre_LCC] +=1
    elif 'green' in LCC_data:
        pre_LCC = 4
        LCC_number[pre_LCC] +=1
    elif 'multicolor' in LCC_data:
        pre_LCC = 5
        LCC_number[pre_LCC] +=1
    elif 'grey' in LCC_data:
        pre_LCC = 6
        LCC_number[pre_LCC] +=1
    elif 'white' in LCC_data:
        pre_LCC = 1
        LCC_number[pre_LCC] +=1
    elif 'yellow' in LCC_data:
        pre_LCC = 8
        LCC_number[pre_LCC] +=1
    elif 'dark brown' in LCC_data:
        pre_LCC = 9
        LCC_number[pre_LCC] +=1
    elif 'purple' in LCC_data:
        pre_LCC = 10
        LCC_number[pre_LCC] +=1
    elif 'pink' in LCC_data:
        pre_LCC = 11
        LCC_number[pre_LCC] +=1
    elif 'n/a' or 'no' in LCC_data:
        pre_LCC = 0
        LCC_number[pre_LCC] +=1
    if pre_LCC == q_lower_color:
        counts['lower_color'] += 1
    LCS_data = data['pre_label_LCS'].lower()
    if 'long' in LCS_data:
        pre_LCS = 1
        LCS_number[pre_LCS] +=1
    elif 'short' in LCS_data:
        pre_LCS = 2
        LCS_number[pre_LCS] +=1
    elif 'skirt' in LCS_data:
        pre_LCS = 3
        LCS_number[pre_LCS] +=1
    elif 'n/a' or 'no' in LCS_data:
        pre_LCS = 4
        LCS_number[pre_LCS] +=1
    if pre_LCS == q_lower_style:
        counts['lower_style'] +=1 
counts['total_gender_number'] = sum(gender_number)
counts['total_backpack_number'] = sum(backpack_number)
counts['total_hat_number'] = sum(hat_number)
counts['total_UCC_number'] = sum(UCC_number)
counts['total_UCS_number'] = sum(UCS_number)
counts['total_LCC_number'] = sum(LCC_number)
counts['total_LCS_number'] = sum(LCS_number)


# for key, value in counts.items():
#     print('{:<50} : {}'.format(key, value))

# 创建表格对象
table1 = PrettyTable()
table1.field_names = ['Category', 'Count']

# 遍历 counts 字典的键值对，将数据添加到表格中
for category, count in counts.items():
    table1.add_row([category, count])

# 打印表格
print(table1)
names = ["gender", "backpack", "hat", "upper_color", "upper_style","lower_color",'lower_style']
accuracy_per_attribute = []
for name in names:
    acc = counts[name] / counts['total_number']
    accuracy_per_attribute.append(acc)
table = PrettyTable(["task", "gender", "backpack", "hat", "upper_color", "upper_style","lower_color",'lower_style'])
formatted_accuracy_per_attribute = ["{:.2%}".format(accuracy) for accuracy in accuracy_per_attribute]
table.add_row(["Attribute Recognition"] + formatted_accuracy_per_attribute)
print(table)
print("=====attribute recognition accuracy: {:.2%}=====".format(sum(accuracy_per_attribute)))
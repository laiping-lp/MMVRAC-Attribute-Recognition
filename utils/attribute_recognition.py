import json
import shutil
import os
from PIL import Image
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import openpyxl


def Attribute_Recognition(cfg,attributes,attr_classes,data_list,gen_attr_reslut = False):
    print("=> Attribute Recognition")
    num_attributes = 7  # num_class of attributes
    accuracy_per_attribute = [0] * num_attributes  # init attribute accuracy list
    count_per_attribute = [0] * num_attributes  # init attributes count list
    attr_re_path = [[] for _ in range(7)]
    attr_label = [[] for _ in range(7)]
    attr_pre_label = [[] for _ in range(7)]
    for i in range(len(attributes)):
        informations = data_list[i // 7]
        for j in range(len(attributes[i])):
            if attr_classes[i][j] == attributes[i][j]:
                accuracy_per_attribute[i % 7] += 1
                # all_result
                # attr_re_path[i % 7].append(informations['img_path'][j])
                # attr_pre_label[i % 7].append(attr_classes[i][j])
                # attr_label[i % 7].append(attributes[i][j])
            else:
                attr_re_path[i % 7].append(informations['img_path'][j])
                attr_pre_label[i % 7].append(attr_classes[i][j])
                attr_label[i % 7].append(attributes[i][j])
            count_per_attribute[i % 7] += 1   
    for j in range(num_attributes):
        accuracy_per_attribute[j] /= count_per_attribute[j]
    attr_dict = {
        "gender" : 0, 
        "backpack":1,
        "hat":2, 
        "upper_color":3,
        "upper_style":4,
        "lower_color":5,
        'lower_style':6
    }
    # names = ["gender", "backpack", "hat", "upper_color", "upper_style","lower_color",'lower_style']
    # names = ['gender']
    names = ['backpack']
    # names = ['hat']
    # names = ['upper_color']
    # names = ['upper_style']
    # names = ['lower_color']
    # names = ['lower_style']
    if gen_attr_reslut:
        for name in names:
            # gen_attr_re_img(cfg,name,attr_re_path[attr_dict[name]],attr_label[attr_dict[name]],attr_pre_label[attr_dict[name]])
            make_attr_json(cfg,name,attr_re_path[attr_dict[name]],attr_label[attr_dict[name]],attr_pre_label[attr_dict[name]])
            # break
            # break
        make_attr_error_type_xlsx(cfg)
    return accuracy_per_attribute

def gen_attr_re_img(cfg,name,attr_re_path,attr_label,attr_pre_label):
    print("=> Attribute Recognition error image generate")
    genders = ["male","female"]
    backpacks = ["n/a","red","black","green","yellow","n/a"]
    hats = ["n/a","red","black","yellow","white","n/a"]
    upperclothing_colors = ['n/a', 'red', 'black', 'blue', 'green', 'multicolor', 'grey', 'white', 'yellow', 'dark brown', 'purple', 'pink']
    upperclothing_styles = ['n/a', 'long', 'short', 'skirt']
    lowerclothing_colors = ['n/a', 'red', 'black', 'blue', 'green', 'multicolor', 'grey', 'white', 'yellow', 'dark brown', 'purple', 'pink']
    lowerclothing_styles = ['n/a', 'long', 'short', 'skirt']
    attr_dict = {
        "gender" : genders, 
        "backpack":backpacks,
        "hat":hats, 
        "upper_color":upperclothing_colors,
        "upper_style":upperclothing_styles,
        "lower_color":lowerclothing_colors,
        'lower_style':lowerclothing_styles,
    }
    image_paths = []  
    pre_matchs = []
    real_matchs = []
    save_folder = cfg.LOG_ROOT + cfg.LOG_NAME + f"/img/{name}_img"
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    # bulid
    else:
        os.makedirs(save_folder)
    for i,(path,real_label,pre_label) in enumerate(zip(attr_re_path,attr_label,attr_pre_label)):
        image_paths.append(path)
        pre_matchs.append(pre_label)
        real_matchs.append(real_label)
        if(i % 10 == 9):
            fig, axs = plt.subplots(1,10, figsize=(12,3))
            for j,image_path in enumerate(image_paths):
                image = Image.open(image_path)
                axs[j].imshow(image)
                axs[j].axis('off')
                axs[j].set_title(f"{attr_dict[name][pre_matchs[j]]}_{attr_dict[name][real_matchs[j]]}",color = 'green')
                # axs[j].set_title(0.5,-1.15,genders[matchs[j]], color = 'red',fontsize=12, ha='center', va='center',transform=axs[j].transAxes)
            image_paths = [] 
            pre_matchs = [] 
            real_matchs = [] 
            plt.tight_layout()
            plt.savefig(f"{save_folder}/{i}.jpg")
            plt.close()
            # break
    
def make_attr_json(cfg,name,attr_re_path,attr_label,attr_pre_label):
    print(f"=> Attribute Recognition error {name} case json generate")
    load_data = []
    for (path,real_label,pre_label) in zip(attr_re_path,attr_label,attr_pre_label):
        data = {
            "file_path":"",
            "pre_label":"",
            "real_label":"" 
            }
        data["file_path"] = path
        data['real_label'] = real_label.item()
        data['pre_label'] = pre_label
        load_data.append(data)
    
    # save_folder = cfg.LOG_ROOT + cfg.LOG_NAME + "/attr_all_result_json_file"
    save_folder = cfg.LOG_ROOT + cfg.LOG_NAME + "/json_file"
    if  not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print(len(load_data))
    with open(f"{save_folder}/{name}.json",'w') as f:
        json.dump(load_data,f)

def make_attr_error_type_xlsx(cfg):
    print("=> Attribute error type generate xlsx!")
    genders = ["male","female"]
    backpacks = ["n/a","red","black","green","yellow","n/a"]
    hats = ["n/a","red","black","yellow","white","n/a"]
    upperclothing_colors = ['n/a', 'red', 'black', 'blue', 'green', 'multicolor', 'grey', 'white', 'yellow', 'dark brown', 'purple', 'pink']
    upperclothing_styles = ['n/a', 'long', 'short', 'skirt']
    lowerclothing_colors = ['n/a', 'red', 'black', 'blue', 'green', 'multicolor', 'grey', 'white', 'yellow', 'dark brown', 'purple', 'pink']
    lowerclothing_styles = ['n/a', 'long', 'short', 'skirt']
    attr_dict = {
        "gender" : genders, 
        "backpack":backpacks,
        "hat":hats, 
        "upper_color":upperclothing_colors,
        "upper_style":upperclothing_styles,
        "lower_color":lowerclothing_colors,
        'lower_style':lowerclothing_styles,
    }
    # names = ['gender','backpack','hat','lower_color','lower_style',"upper_color",'upper_style']
    # names = ['gender']
    names = ['backpack']
    # names = ['hat']
    # names = ['upper_color']
    # names = ['upper_style']
    # names = ['lower_color']
    # names = ['lower_style']
    dir_path = cfg.LOG_ROOT + cfg.LOG_NAME + "/json_file"
    for name in names:
        file_path = f"{dir_path}/{name}.json"
        with open(file_path,'r') as f:
            load_data = json.load(f)
        label_list = set()
        for data in tqdm(load_data):
            pre = data['pre_label']
            real = data['real_label']
            label = attr_dict[name][pre]+ "_" + attr_dict[name][real]
            # label = str(pre)+ "_" + str(real)
            # print(label)
            # break
            label_list.add(label)
        # print(len(label_list))
        # print(label_list)
        label_list = list(label_list)
        label_list_count = []
        for i in range(len(label_list)):
            label_list_count.append(0)
        for i in range(len(label_list)):
            for data in (load_data):
                pre = data['pre_label']
                real = data['real_label']
                label = attr_dict[name][pre]+ "_" + attr_dict[name][real]
                # label = str(pre)+ "_" + str(real)
                if(label == label_list[i]):
                    label_list_count[i] += 1
        # print(label_list_count)
        save_folder = cfg.LOG_ROOT + cfg.LOG_NAME + "/error_type_xlsx"
        if  not os.path.exists(save_folder):
            os.makedirs(save_folder)
        filename = f'{save_folder}/{name}.xlsx'
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['pre_real', 'number'])  # 写入表头

        for i in range(len(label_list)):
            ws.append([label_list[i], label_list_count[i]])

        wb.save(filename)
        # with open(filename, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['pre_real', 'number'])  # 写入表头
        #     for i in range(len(label_list)):
        #         writer.writerow([label_list[i], label_list_count[i]])
        # break
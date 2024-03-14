import json
import shutil
import os
from PIL import Image
import matplotlib.pyplot as plt


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
    names = ["gender", "backpack", "hat", "upper_color", "upper_style","lower_color",'lower_style']
    if gen_attr_reslut:
        for name in names:
            gen_attr_re_img(cfg,name,attr_re_path[attr_dict[name]],attr_label[attr_dict[name]],attr_pre_label[attr_dict[name]])
            make_attr_json(cfg,name,attr_re_path[attr_dict[name]],attr_label[attr_dict[name]],attr_pre_label[attr_dict[name]])
            # break
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
    save_folder = cfg.LOG_ROOT + cfg.LOG_NAME + f"/{name}_img"
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
    print("=> Attribute Recognition error case json generate")
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
    
    save_folder = save_folder = cfg.LOG_ROOT + cfg.LOG_NAME + f"/{name}.json"
    with open(save_folder,'w') as f:
        json.dump(load_data,f)

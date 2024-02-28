import json
import os
import re

file_path = "/data3/laiping/exp/uavhuman_humanbench_vit_large/cases.json"
with open(file_path,'r') as f:
    load_data = json.load(f)
    
counts = {
    'bad_case_number': 0,
    'pid_not_match_number': 0,
    'pid_match_setupid_not_match_number': 0,
    'gender_not_match': 0,
    'hat_not_match': 0,
    'backpack_not_match': 0,
    'UC_not_match': 0,
    'UC_color_not_match': 0,
    'UC_style_not_match': 0,
    'LC_not_match': 0,
    'LC_color_not_match': 0,
    'LC_style_not_match': 0,
    'action_not_match': 0
}
pattern_gender = re.compile(r'G([-\d]+)')
pattern_backpack = re.compile(r'B([-\d]+)')
pattern_hat = re.compile(r'H([-\d]+)')
pattern_upper = re.compile(r'UC([-\d]+)')
pattern_lower = re.compile(r'LC([-\d]+)')
pattern_action = re.compile(r'A([-\d]+)')
for data in load_data:
    query_id = data['query_id']
    query_filename = os.path.basename(data["query_path"])
    query_pid = query_id // 100
    query_setupid = query_id % 100
    q_gender = int(pattern_gender.search(query_filename).groups()[0][0]) # 0: n/a; 1: male; 2: female
    q_backpack = int(pattern_backpack.search(query_filename).groups()[0][0]) # 0: n/a; 1: red; 2: black; 3: green; 4: yellow; 5: n/a
    q_hat = int(pattern_hat.search(query_filename).groups()[0][0]) # 0: n/a; 1: red; 2: black; 3: yellow; 4: white; 5: n/a
    q_upper_cloth = pattern_upper.search(query_filename).groups()[0][:3]
    q_upper_color = int(q_upper_cloth[:2]) # 0: n/a; 1: red; 2: black; 3: blue; 4: green; 5: multicolor; 6: grey; 7: white; 8: yellow; 9: dark brown; 10: purple; 11: pink
    q_upper_style = int(q_upper_cloth[2]) # 0: n/a; 1: long; 2: short; 3: skirt
    q_lower_cloth = pattern_lower.search(query_filename).groups()[0][:3]
    q_lower_color = int(q_lower_cloth[:2]) # 0: n/a; 1: red; 2: black; 3: blue; 4: green; 5: multicolor; 6: grey; 7: white; 8: yellow; 9: dark brown; 10: purple; 11: pink
    q_lower_style = int(q_lower_cloth[2]) # 0: n/a; 1: long; 2: short; 3: skirt
    q_action = int(pattern_action.search(query_filename).groups()[0])
    top_50_results_ids = data["top_50_results_ids"]
    top_50_results_path = data["top_50_results_path"]
    if query_id != top_50_results_ids[0]:
        counts["bad_case_number"] += 1
        for j,(id,path) in enumerate(zip(top_50_results_ids,top_50_results_path)):
            filename = os.path.basename(path) 
            g_gender = int(pattern_gender.search(filename).groups()[0][0]) # 0: n/a; 1: male; 2: female
            g_backpack = int(pattern_backpack.search(filename).groups()[0][0]) # 0: n/a; 1: red; 2: black; 3: green; 4: yellow; 5: n/a
            g_hat = int(pattern_hat.search(filename).groups()[0][0]) # 0: n/a; 1: red; 2: black; 3: yellow; 4: white; 5: n/a
            g_upper_cloth = pattern_upper.search(filename).groups()[0][:3]
            g_upper_color = int(g_upper_cloth[:2]) # 0: n/a; 1: red; 2: black; 3: blue; 4: green; 5: multicolor; 6: grey; 7: white; 8: yellow; 9: dark brown; 10: purple; 11: pink
            g_upper_style = int(g_upper_cloth[2]) # 0: n/a; 1: long; 2: short; 3: skirt
            g_lower_cloth = pattern_lower.search(filename).groups()[0][:3]
            g_lower_color = int(g_lower_cloth[:2]) # 0: n/a; 1: red; 2: black; 3: blue; 4: green; 5: multicolor; 6: grey; 7: white; 8: yellow; 9: dark brown; 10: purple; 11: pink
            g_lower_style = int(g_lower_cloth[2]) # 0: n/a; 1: long; 2: short; 3: skirt
            g_action = int(pattern_action.search(filename).groups()[0])
            if(j == 50):
                break
            pid = id // 100
            setupid = id % 100
            if(pid != query_pid):
                counts["pid_not_match_number"] += 1
            else:
                counts["pid_match_setupid_not_match_number"] += 1
                if(q_gender != g_gender):
                    counts['gender_not_match'] += 1
                if(q_hat != g_hat):
                    counts['hat_not_match'] += 1
                if(q_backpack != g_backpack):
                    counts['backpack_not_match'] += 1
                if(q_upper_cloth != g_upper_cloth):
                    counts['UC_not_match'] += 1
                if(q_lower_cloth != g_lower_cloth):
                    counts["LC_not_match"] += 1
                if(q_upper_color != g_upper_color):
                    counts['UC_color_not_match'] += 1
                if(q_upper_style != g_upper_style):
                    counts['UC_style_not_match'] += 1
                if(q_lower_color != g_lower_color):
                    counts['LC_color_not_match'] += 1
                if(q_lower_style != g_lower_style):
                    counts['LC_style_not_match'] += 1
                if(q_action != g_action):
                    counts['action_not_match'] += 1            
            # break
    # break
print(counts)
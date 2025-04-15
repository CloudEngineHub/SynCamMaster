import json
import numpy as np
import os

root = "./syncam_train_data/train/f24_aperture5"
with open('./syncam_train_data/train/rotation_data.json', 'r') as f:
    data = json.load(f)

az_diff_max = 30
el_diff_max = 15
for scene_num in range(1, 3401):
    scene_key = f'scene{scene_num}'
    scene_availble_list = []
    for i in range(1, 11):
        for j in range(1, 11):
            if i != j:
                az1, el1 = data[scene_key][f"traj{i}"]["rz"], data[scene_key][f"traj{i}"]["ry"]
                az2, el2 = data[scene_key][f"traj{j}"]["rz"], data[scene_key][f"traj{j}"]["ry"]
                if abs(az1 - az2) < az_diff_max and abs(el1 - el2) < el_diff_max:
                    scene_availble_list.append((i, j))
    folder_name = f"scene{scene_num}"
    np.save(os.path.join(root, folder_name, f"2view_az{az_diff_max}_el{el_diff_max}_available_list.npy"), scene_availble_list)
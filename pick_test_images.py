from __future__ import absolute_import, division, print_function


import os
import shutil
from tqdm import tqdm

os.makedirs("./fortest/data", exist_ok=True)

root = "Your_path_to_kitti_raw"

with open("./splits/eigen/test_files.txt","r") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        folder, idx, d = line[:-1].split(' ')
        lr = "image_02" if d=="l" else "image_03"

        path = f"{root}/{folder}/{lr}/data/{idx}.jpg"

        assert os.path.isfile(path)

        new_name = f"{folder.split('/')[-1]}_{lr}_{idx}.jpg"

        shutil.copyfile(path, f"./fortest/data/{new_name}")

    f.close()

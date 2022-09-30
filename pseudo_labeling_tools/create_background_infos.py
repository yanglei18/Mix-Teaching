import os
import sys
import argparse
import pickle

import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

def parse_option():
    parser = argparse.ArgumentParser('filter unauthentic instances from database based on confidence and uncertainty', add_help=False)
    parser.add_argument('--kitti_root', type=str, required=False, metavar="", help='path to kitti dataset', )
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_option()
    kitti_root = args.kitti_root
    with open(os.path.join(kitti_root, "kitti_infos_test.pkl"), 'rb') as f:
        kitti_infos = pickle.load(f)
    print(len(kitti_infos))
    kitti_background_infos = []
    for idx in tqdm(range(len(kitti_infos))):
        info = kitti_infos[idx]
        annos = info["annos"]

        if isinstance(annos["name"], list):
            continue
        num_obj = np.sum(annos["index"] >= 0)
        if num_obj == 0:
            kitti_background_infos.append(info)
    print(len(kitti_background_infos))
    with open(os.path.join(kitti_root, "kitti_infos_background.pkl"), 'wb') as f:
        pickle.dump(kitti_background_infos, f)

        



import os
import cv2
import math
import pathlib
import argparse
import pickle

import kitti_common as kitti

import numpy as np
from tqdm import tqdm

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Van': 3,
    'Person_sitting': 4,
    'Truck': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': -1,
}

def parse_option():
    parser = argparse.ArgumentParser('Convert kitti raw data to standard kitti format', add_help=False)
    parser.add_argument('--kitti_root', type=str, required=False, metavar="", help='path to kitti det root', )
    parser.add_argument('--ssl', type=bool, required=False, metavar="", help='path to kitti det root', )
    args = parser.parse_args()
    return args

def convertRot2Alpha(ry3d, z3d, x3d):
    alpha = ry3d - math.atan2(x3d, z3d)
    while alpha > math.pi: alpha -= math.pi * 2
    while alpha < (-math.pi): alpha += math.pi * 2
    return alpha

def visualization(img, anno):
    dim = anno["dim"]
    loc = anno["loc"]
    roty = anno["roty"]
    P2 = anno["P2"]
    box3d = kitti.compute_box_3d_image(P2, roty, dim, loc)
    img = kitti.draw_box_3d(img, box3d)
    return img

def encode_bbox(P, ry, dims, locs, img_shape):
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]
    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]
    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2
    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])
    corners_3d_extend = corners_3d.transpose(1, 0)
    corners_3d_extend = np.concatenate(
        [corners_3d_extend, np.ones((corners_3d_extend.shape[0], 1), dtype=np.float32)], axis=1)    
    corners_2d = np.matmul(P, corners_3d_extend.transpose(1, 0))
    corners_2d = corners_2d[:2] / corners_2d[2]
    bbox = np.array([min(corners_2d[0]), min(corners_2d[1]),
                     max(corners_2d[0]), max(corners_2d[1])])

    bbox[[0,2]] = np.clip(bbox[[0,2]], 0, img_shape[1])
    bbox[[1,3]] = np.clip(bbox[[1,3]], 0, img_shape[0])
    return bbox

def _read_imageset_file(kitti_root, path):
    imagetxt = os.path.join(kitti_root, path)
    with open(imagetxt, 'r') as f:
        lines = f.readlines()
    total_img_ids = [int(line) for line in lines]
    img_ids = []
    for img_id in total_img_ids:
        if "test" in path:
            img_path = os.path.join(kitti_root, "testing/image_2", "{:06d}".format(img_id) + ".png")
        else:
            img_path = os.path.join(kitti_root, "training/image_2", "{:06d}".format(img_id) + ".png")
        if os.path.exists(img_path):
            img_ids.append(img_id)
    return img_ids

def create_kitti_info_file(kitti_root,
                           ssl=False,
                           info_path=None,
                           create_trainval=True,
                           relative_path=True):
    train_img_ids = _read_imageset_file(kitti_root, "ImageSets/train.txt")
    val_img_ids = _read_imageset_file(kitti_root, "ImageSets/val.txt")
    trainval_img_ids = _read_imageset_file(kitti_root, "ImageSets/trainval.txt")
    test_img_ids = _read_imageset_file(kitti_root, "ImageSets/test.txt")
    print("Generate info. this may take several minutes.")

    if info_path is None:
        info_path = pathlib.Path(kitti_root)
    else:
        info_path = pathlib.Path(info_path)
    info_path.mkdir(parents=True, exist_ok=True)
    if not ssl:
        kitti_infos_train = kitti.get_kitti_image_info(
            kitti_root,
            training=True,
            label_info=True,
            calib=True,
            image_ids=train_img_ids,
            relative_path=relative_path)
        filename = info_path / 'kitti_infos_train.pkl'
        print(f"Kitti info train file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_train, f)
        
        kitti_infos_val = kitti.get_kitti_image_info(
            kitti_root,
            training=True,
            label_info=True,
            calib=True,
            image_ids=val_img_ids,
            relative_path=relative_path)
        filename = info_path / 'kitti_infos_val.pkl'
        print(f"Kitti info val file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_val, f)

        if create_trainval:
            kitti_infos_trainval = kitti.get_kitti_image_info(
                kitti_root,
                training=True,
                label_info=True,
                calib=True,
                image_ids=trainval_img_ids,
                relative_path=relative_path)
            filename = info_path / 'kitti_infos_trainval.pkl'
            print(f"Kitti info trainval file is saved to {filename}")
            with open(filename, 'wb') as f:
                pickle.dump(kitti_infos_trainval, f)
        
        kitti_infos_test = kitti.get_kitti_image_info(
            kitti_root,
            training=False,
            label_info=False,
            calib=True,
            image_ids=test_img_ids,
            relative_path=relative_path)
        filename = info_path / 'kitti_infos_test.pkl'
        print(f"Kitti info val file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_test, f)
    else:
        kitti_infos_test = kitti.get_kitti_image_info(
            kitti_root,
            training=False,
            label_info=True,
            calib=True,
            image_ids=test_img_ids,
            relative_path=relative_path)
        filename = info_path / 'kitti_infos_test.pkl'
        print(f"Kitti info val file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_test, f)

def create_groundtruth_database(kitti_root,
                                info_path=None,
                                used_classes=["Car", "Pedestrian", "Cyclist"],
                                database_save_path=None,
                                relative_path=True):
    root_path = pathlib.Path(kitti_root)
    if info_path is None:
        info_save_path = root_path / 'kitti_infos_test.pkl'
    else:
        path_info = pathlib.Path(info_path)
        path_info.mkdir(parents=True, exist_ok=True)
        info_save_path = path_info / 'kitti_infos_test.pkl'
    if database_save_path is None:
        database_save_path = root_path / 'gt_database'
    else:
        database_save_path = pathlib.Path(database_save_path)
    database_save_path.mkdir(parents=True, exist_ok=True)

    if info_path is None:
        db_info_save_path = root_path / "kitti_dbinfos_test.pkl"
    else:
        db_info_save_path = path_info / "kitti_dbinfos_test.pkl"
    with open(info_save_path, 'rb') as f:
        kitti_infos = pickle.load(f)

    all_db_infos = {}
    if used_classes is None:
        used_classes = list(kitti.get_classes())
        used_classes.pop(used_classes.index('DontCare'))
    for name in used_classes:
        all_db_infos[name] = {}
        all_db_infos[name] = {}

    for idx in tqdm(range(len(kitti_infos))):
        info = kitti_infos[idx]
        img_path_l = os.path.join(kitti_root, info["img_path"])

        image_idx = info["image_idx"]
        annos = info["annos"]
        if isinstance(annos["name"], list):
            continue
        names = annos["name"] 
        alphas = annos["alpha"]
        dimensions = annos["dimensions"]
        locations = annos["location"]
        rotys = annos["rotation_y"]
        difficulty = annos["difficulty"]
        truncated = annos["truncated"]
        occluded = annos["occluded"]
        scores = annos["score"]
        gt_idxes = annos["index"]
        num_obj = np.sum(annos["index"] >= 0)
        if num_obj == 0:
            continue
        img_l = cv2.imread(img_path_l)
        img_shape = img_l.shape
        img_shape_key = f"{img_shape[0]}_{img_shape[1]}"

        for i in range(num_obj):
            if difficulty[i] == -1:
                continue
            if names[i] not in used_classes:
                continue
            box2d_l = encode_bbox(info['calib/P2'], rotys[i], dimensions[i], locations[i], img_shape)
            cropImg_l = img_l[int(box2d_l[1]):int(box2d_l[3]), int(box2d_l[0]):int(box2d_l[2]), :]
            filename_key = f"{image_idx}_{names[i]}_{gt_idxes[i]}_{difficulty[i]}"
            folder_l = os.path.join(database_save_path, "image_2")
            if not os.path.exists(folder_l):
                os.makedirs(folder_l)
            filepath_l = os.path.join(folder_l, filename_key + ".jpg")
            if cropImg_l.shape[0] == 0 or cropImg_l.shape[1] == 0:
                continue
            cv2.imwrite(filepath_l, cropImg_l)
            
            sample_info = {
                "filename_key": filename_key,
                "name": names[i],
                "label": TYPE_ID_CONVERSION[names[i]],
                "bbox_l": box2d_l,
                "alpha": alphas[i],
                "roty": rotys[i],
                "dim": dimensions[i],
                "loc": locations[i],
                "P2": info['calib/P2'],
                "P3": info['calib/P3'],
                "img_shape": img_shape,
                "image_idx": image_idx,
                "gt_idx": gt_idxes[i],
                "difficulty": difficulty[i],
                "truncated": truncated[i],
                "occluded": occluded[i],
                "filepath_l": filepath_l,
                "score": 1.0
            }
            if "score" in annos:
                sample_info["score"] = annos["score"][i]
            if "geo_conf" in annos:
                sample_info["geo_conf"] = annos["geo_conf"][i]
            if img_shape_key not in all_db_infos[names[i]]:
                all_db_infos[names[i]][img_shape_key] = []
            all_db_infos[names[i]][img_shape_key].append(sample_info)
    for class_key, class_db_infos in all_db_infos.items():
        for k, v in class_db_infos.items():
            print(f"load {len(v)} {k}_{class_key} database infos")
    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

if __name__ == "__main__":
    args = parse_option()
    create_kitti_info_file(args.kitti_root, args.ssl)
    if args.ssl:
        create_groundtruth_database(args.kitti_root)
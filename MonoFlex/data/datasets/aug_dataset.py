import cv2
import os
import pickle
import random
import numpy as np

from random import sample
from PIL import Image
from data.datasets import kitti_common as kitti

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

class AUGDataset():
    def __init__(self, cfg, kitti_root, is_train=True, split="train"):
        super(AUGDataset, self).__init__()
        self.kitti_root = kitti_root
        self.split = split
        self.is_train = is_train
        self.max_objs = cfg.DATASETS.MAX_OBJECTS
        self.classes = cfg.DATASETS.DETECT_CLASSES
        self.use_border_cut = cfg.DATASETS.USE_BORDER_CUT
        self.use_color_padding = cfg.DATASETS.USE_COLOR_PADDING
        self.use_mixup = cfg.DATASETS.USE_MIXUP

        self.cls_threshold = cfg.DATASETS.CLS_THRESHOLD
        self.geo_conf_threshold = cfg.DATASETS.GEO_CONF_THRESHOLD
        self.use_mix_teaching = cfg.DATASETS.USE_MIX_TEACHING 
        self.labeled_ratio = cfg.DATASETS.LABELED_RATIO

        if self.split == "train":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_train.pkl")
            if self.use_mix_teaching:
                background_info_path = os.path.join(self.kitti_root, "../kitti_infos_background.pkl")
                db_info_path = os.path.join(self.kitti_root, "../kitti_dbinfos_test_filtered.pkl")
        elif self.split == "val":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_val.pkl")
        elif self.split == "trainval":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_trainval.pkl")
            db_info_path = os.path.join(self.kitti_root, "../kitti_dbinfos_test_filtered.pkl")
        elif self.split == "test":
            info_path = os.path.join(self.kitti_root, "../kitti_infos_test.pkl")
        else:
            raise ValueError("Invalid split!")

        with open(info_path, 'rb') as f:
            kitti_infos = pickle.load(f)
            if self.split in ["train"]: 
                self.kitti_infos = kitti_infos[:int(self.labeled_ratio * len(kitti_infos))]
            else:
                self.kitti_infos = kitti_infos
        self.num_samples = len(self.kitti_infos)

        if self.is_train and self.use_mix_teaching:
            with open(background_info_path, 'rb') as f:
                self.kitti_background_infos = pickle.load(f)
            with open(db_info_path, 'rb') as f:
                self.db_infos = pickle.load(f)

            self.sample_nums_hashmap = dict()
            self.sample_counter = dict()
            for class_name, class_db_infos in self.db_infos.items():
                self.sample_nums_hashmap[class_name] = {}
                self.sample_counter[class_name] = {}
                for img_shape_key, class_shape_db_infos in class_db_infos.items():
                    self.sample_nums_hashmap[class_name][img_shape_key] = len(class_shape_db_infos)
                    self.sample_counter[class_name][img_shape_key] = 0
        
        self.class_aug_nums = {"Car": 24, "Pedestrian": 12, "Cyclist": 12}
        self.frame_cache = None

    def reset_sample_counter(self):
        for class_name, class_counter in self.sample_counter.items():
            for img_shape_key, class_shape_counter in class_counter.items():
                self.sample_counter[class_name][img_shape_key] = 0

    def update_sample_counter(self, aug_class, img_shape_key):
        total_nums = self.sample_nums_hashmap[aug_class][img_shape_key]
        current_nums = self.sample_counter[aug_class][img_shape_key]
        if current_nums < total_nums - 1:
            self.sample_counter[aug_class][img_shape_key] = current_nums + 1
        else:
            self.sample_counter[aug_class][img_shape_key] = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        info = self.kitti_infos[idx]
        use_bcp = False
        if self.is_train and self.use_mix_teaching and random.random() < 0.420:
            use_bcp = True
            info = random.choice(self.kitti_background_infos)
        
        image_idx = info["image_idx"]
        img_path = os.path.join(self.kitti_root, "../" + info["img_path"])
        img = cv2.imread(img_path)
        
        P = P2 = info["calib/P2"]
        if not self.is_train:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return img, P2, image_idx
        
        annos = info["annos"]
        names = annos["name"] 
        bboxes = annos["bbox"]
        alphas = annos["alpha"]
        dimensions = annos["dimensions"]
        locations = annos["location"]
        rotys = annos["rotation_y"]
        difficulty = annos["difficulty"]
        truncated = annos["truncated"]
        occluded = annos["occluded"]
        scores = annos["score"]

        embedding_annos = []
        init_bboxes = []        
        for i in range(len(names)):
            init_bboxes.append(bboxes[i])
            if names[i] not in self.classes:
                continue
            ins_anno = {
                    "name": names[i],
                    "label": TYPE_ID_CONVERSION[names[i]],
                    "bbox": bboxes[i],
                    "alpha": alphas[i],
                    "dim": dimensions[i],
                    "loc": locations[i],
                    "roty": rotys[i],
                    "P": P2,
                    "difficulty": difficulty[i],
                    "truncated": truncated[i],
                    "occluded": occluded[i],
                    "flipped": False,
                    "score": scores[i],
                    "gt_label": True
                }
            embedding_annos.append(ins_anno)
        init_bboxes = np.array(init_bboxes)

        use_bcp = True if self.use_mix_teaching and random.random() < 0.5 else use_bcp
        if use_bcp:
            ori_annos_num = len(embedding_annos)
            for aug_class, aug_nums in self.class_aug_nums.items():
                img_shape_key = f"{img.shape[0]}_{img.shape[1]}"
                if img_shape_key in self.db_infos[aug_class].keys():
                    class_db_infos = self.db_infos[aug_class][img_shape_key]
                    if len(class_db_infos) == 0:
                        continue
                    trial_num = aug_nums + 60
                    nums = 0
                    for i in range(trial_num):
                        if nums >= aug_nums:
                            break
                        sample_id = self.sample_counter[aug_class][img_shape_key]
                        self.update_sample_counter(aug_class, img_shape_key) 
                        ins = class_db_infos[sample_id]
                        patch_img_path = os.path.join(self.kitti_root, "../", ins["filepath_l"])
                        box2d = ins["bbox_l"]
                        P = ins["P2"]
                        
                        if ins['difficulty'] > 0:
                            continue
                        if ins['score'] < self.cls_threshold or ins['geo_conf'] < self.geo_conf_threshold:
                            continue
                        if len(init_bboxes.shape) > 1:
                            ious = kitti.iou(init_bboxes, box2d[np.newaxis, ...])
                            if np.max(ious) > 0.0:
                                continue
                            init_bboxes = np.vstack((init_bboxes, box2d[np.newaxis, ...]))
                        else:
                            init_bboxes = box2d[np.newaxis, ...].copy()
                        
                        patch_img = cv2.imread(patch_img_path)
                        box2d_h = int(box2d[3]) - int(box2d[1])
                        box2d_w = int(box2d[2]) - int(box2d[0])
                        img_ind_0, img_ind_1 = int(box2d[0]), int(box2d[1])
                        patch_img_ind_0, patch_img_ind_1 = 0, 0
                        if ins["name"] == "Car" and self.use_border_cut:
                            box2d_w_delta = int(0.2 * random.random() * box2d_w)
                            box2d_h_delta = int(0.2 * random.random() * box2d_h)
                            box2d_w = box2d_w - box2d_w_delta
                            box2d_h = box2d_h - box2d_h_delta
                            if random.random() < 0.5:
                                img_ind_1 = int(box2d[1]) + box2d_h_delta
                                patch_img_ind_1 = box2d_h_delta
                            if random.random() < 0.5:
                                img_ind_0 = int(box2d[0]) + box2d_w_delta
                                patch_img_ind_0 = box2d_w_delta
                        
                        if random.random() < 0.5 and self.use_color_padding:
                            if patch_img_ind_0 > 0:
                                img[int(box2d[1]):int(box2d[3]):, int(box2d[0]): img_ind_0, :] = [random.randint(64, 191) for _ in range(3)]
                            else:
                                img[int(box2d[1]):int(box2d[3]), img_ind_0 + box2d_w :int(box2d[2]), :] = [random.randint(64, 191) for _ in range(3)]
                            if patch_img_ind_1 > 0:
                                img[int(box2d[1]): img_ind_1, int(box2d[0]):int(box2d[2]), :] = [random.randint(64, 191) for _ in range(3)]
                            else:
                                img[img_ind_1 + box2d_h:int(box2d[3]), int(box2d[0]):int(box2d[2]), :] = [random.randint(64, 191) for _ in range(3)]

                        ratio = random.randint(6, 10) / 10 if self.use_mixup else 1.0
                        img[img_ind_1 : img_ind_1 + box2d_h, img_ind_0 : img_ind_0 + box2d_w, :] =\
                            img[img_ind_1 : img_ind_1 + box2d_h, img_ind_0 : img_ind_0 + box2d_w, :] * (1 - ratio) +\
                            patch_img[patch_img_ind_1 : patch_img_ind_1 + box2d_h, patch_img_ind_0 : patch_img_ind_0 + box2d_w, :] * ratio

                        ins_anno = {
                            "name": ins["name"],
                            "label": TYPE_ID_CONVERSION[ins["name"]],
                            "bbox": box2d,
                            "alpha": ins["alpha"],
                            "dim": ins["dim"],
                            "loc": ins["loc"],
                            "roty": ins["roty"],
                            "P": P,
                            "difficulty": ins["difficulty"],
                            "truncated": ins["truncated"],
                            "occluded": ins["occluded"],
                            "flipped": False,
                            "score": ins["score"],
                            "gt_label": False
                        }
                        embedding_annos.append(ins_anno)
                        nums += 1
            aug_annos_num = len(embedding_annos)
            if ori_annos_num == aug_annos_num:
                use_bcp = False

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if (self.frame_cache is None or random.random() < 0.05) and len(embedding_annos) > 3:
            self.frame_cache = {"img": img, "P": P, "embedding_annos": embedding_annos, "image_idx": image_idx, "use_bcp": use_bcp}
        if len(embedding_annos) == 0 and self.frame_cache is not None:
            img, P, use_bcp, embedding_annos, image_idx = self.frame_cache["img"], self.frame_cache["P"], self.frame_cache["use_bcp"], \
                self.frame_cache["embedding_annos"], self.frame_cache["image_idx"]
        return img, P, use_bcp, embedding_annos, image_idx

    def visualization(self, img, annos, save_path):
        image = img.copy()
        for anno in annos:
            dim = anno["dim"]
            loc = anno["loc"]
            roty = anno["roty"]
            bbox = anno["bbox"]
            box3d = kitti.compute_box_3d_image(anno["P"], roty, dim, loc)
            image = kitti.draw_box_3d(image, box3d)
        cv2.imwrite(save_path, image)
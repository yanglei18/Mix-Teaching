import os
import sys
import argparse
import cv2

import numpy as np
from shutil import copyfile
from tqdm import tqdm

from utils.kitti_utils import *
from utils.utils import Projector, load_intrinsic, load_raw_data_intrinsic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

def parse_option():
    parser = argparse.ArgumentParser('Convert kitti raw data to standard kitti format', add_help=False)
    parser.add_argument('--raw_data_root', type=str, required=False, metavar="", help='path to kitti raw data', )
    parser.add_argument('--kitti_root', type=str, required=False, metavar="", help='path to kitti det root', )
    parser.add_argument('--demo', type=bool, default=False, help='if save demo image',)

    args = parser.parse_args()
    return args

def get_pixel_key(image_name):
    image = cv2.imread(image_name)
    pixel_key = '{:.8f}_{:.8f}'.format(np.mean(image), np.std(image))
    return pixel_key

def load_obj3d_frames(kitti_root):
    image_database = []
    image_path = os.path.join(kitti_root, "training", "image_2")
    for image_name in tqdm(os.listdir(image_path)):
        pixel_key = get_pixel_key(image_name)
        image_database.append(pixel_key)
    return image_database

def calib_generation(lidar_calib_file, cam_calib_file, calib_path):
    vel_to_cam, cam_to_vel, R0_rect, Tr_velo_to_cam = KittiCalibration.get_transform_matrix(lidar_calib_file, cam_calib_file)
    P3, P2 = load_raw_data_intrinsic(cam_calib_file)
    kitti_calib = dict()
    kitti_calib["P0"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P1"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P2"] = P2  # Left camera transform.
    kitti_calib["P3"] = P3  # Dummy values.
    # Cameras are already rectified.
    kitti_calib["R0_rect"] = R0_rect[:3, :3]
    kitti_calib["Tr_velo_to_cam"] = Tr_velo_to_cam[:3, :4]
    kitti_calib["Tr_imu_to_velo"] = np.zeros((3, 4))  # Dummy values.
    with open(calib_path, "w") as calib_file:
        for (key, val) in kitti_calib.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            calib_file.write("%s: %s\n" % (key, val_str))

    return P2, P3, vel_to_cam

def copy_file(file_src, file_dest):
    if not os.path.exists(file_dest):
        try:
            copyfile(file_src, file_dest)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)

if __name__ == "__main__":
    args = parse_option()
    raw_data_root, kitti_root, is_demo = args.raw_data_root, args.kitti_root, args.demo
    image_database = load_obj3d_frames(kitti_root)
    dates = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_10_03', '2011_09_30']
    
    kitti_root = os.path.join(kitti_root, "testing")
    os.makedirs(os.path.join(kitti_root, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(kitti_root, "image_2"), exist_ok=True)
    os.makedirs(os.path.join(kitti_root, "image_3"), exist_ok=True)
    os.makedirs(os.path.join(kitti_root, "calib"), exist_ok=True)

    frame_id = 0
    for date_id in dates:
        date_path = os.path.join(raw_data_root, date_id)
        cam_calib_file = os.path.join(date_path, "calib_cam_to_cam.txt")
        lidar_calib_file = os.path.join(date_path, "calib_velo_to_cam.txt")
        for scene_folder in os.listdir(date_path):
            print(scene_folder)
            if 'txt' in scene_folder:
                continue
            scene_folder = os.path.join(date_path, scene_folder)

            src_image_02_path = os.path.join(scene_folder, "image_02", "data")
            src_image_03_path = os.path.join(scene_folder, "image_03", "data")
            src_velodyne_path = os.path.join(scene_folder, "velodyne_points", "data")

            for file_name in os.listdir(src_image_02_path):
                raw_file_name = file_name.split('.')[0]
                dest_file_name = "{:06d}".format(frame_id)
                frame_id = frame_id + 1
                
                src_image_02_file_name = os.path.join(src_image_02_path, raw_file_name + ".png")
                src_image_03_file_name = os.path.join(src_image_03_path, raw_file_name + ".png")
                src_velodyne_file_name = os.path.join(src_velodyne_path, raw_file_name + ".bin")
                dest_image_02_file_name = os.path.join(kitti_root, "image_2", dest_file_name + ".png")
                dest_image_03_file_name = os.path.join(kitti_root, "image_3", dest_file_name +".png")
                dest_velodyne_file_name = os.path.join(kitti_root, "velodyne", dest_file_name + ".bin")
                dest_calib_file_name = os.path.join(kitti_root, "calib", dest_file_name + ".txt")
    
                if os.path.exists(src_image_02_file_name) and os.path.exists(src_image_03_file_name) and os.path.exists(src_velodyne_file_name):
                    pixel_key = get_pixel_key(src_image_02_file_name)
                    if pixel_key in image_database:
                        continue
                    P2, P3, vel_to_cam = calib_generation(lidar_calib_file, cam_calib_file, dest_calib_file_name)
                    copy_file(src_image_02_file_name, dest_image_02_file_name)
                    copy_file(src_image_03_file_name, dest_image_03_file_name)
                    copy_file(src_velodyne_file_name, dest_velodyne_file_name)

                    if is_demo:
                        projector = Projector("front", P2)
                        image = cv2.imread(dest_image_02_file_name)
                        points = np.fromfile(dest_velodyne_file_name, dtype=np.float32, count=-1).reshape([-1, 4])
                        image = projector.project(points, image, vel_to_cam)
                        cv2.imwrite("demo.jpg", image)


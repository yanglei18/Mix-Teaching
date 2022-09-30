from ast import arg
import os
import argparse

def parse_option():
    parser = argparse.ArgumentParser('generate imageset.txt based on image_2', add_help=False)
    parser.add_argument('--kitti_root', type=str, required=False, metavar="", help='path to kitti_root', )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_option()
    kitti_root = args.kitti_root
    image_2_path = os.path.join(kitti_root, "testing", "image_2")
    imageset_txt = os.path.join(kitti_root, "ImageSets", "test.txt")
    img_ids = []
    for image_name in os.listdir(image_2_path):
        img_id = int(image_name.split('.')[0])
        img_f = os.path.join(image_2_path, image_name)
        if os.path.getsize(img_f) < 10:
            continue
        img_ids.append(img_id)

    with open(imageset_txt,'w') as f:
        for idx in img_ids:
            frame_name = "{:06d}".format(idx)
            f.write(frame_name)
            f.write("\n")
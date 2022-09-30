import argparse
import pickle

def parse_option():
    parser = argparse.ArgumentParser('filter unauthentic instances from database based on confidence and uncertainty', add_help=False)
    parser.add_argument('--old_db_infos', type=str, required=False, metavar="", help='path to old db_infos.pkl', )
    parser.add_argument('--new_db_infos', type=str, required=False, metavar="", help='path to new db_infos.pkl', )
    parser.add_argument('--score_threshold', type=float, default=0.65, help='set score threshold',)
    parser.add_argument('--geo_conf_threshold', type=float, default=0.85, help='set geo_conf threshold',)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_option()
    old_db_infos, new_db_infos  = args.old_db_infos, args.new_db_infos
    score_threshold, geo_conf_threshold =  args.score_threshold, args.geo_conf_threshold
    with open(old_db_infos, 'rb') as f:
        kitti_db_infos = pickle.load(f)
    sub_db_infos = dict()
    total_valid_ins, total_ins = 0, 0
    for class_name, class_db_infos in kitti_db_infos.items():
        if class_name not in sub_db_infos:
            sub_db_infos[class_name] = {}
        for img_shape, shape_dbinfos in class_db_infos.items():
            if img_shape not in sub_db_infos[class_name]:
                sub_db_infos[class_name][img_shape] = []
            geo_count = 0
            for ins_idx in range(len(shape_dbinfos)):
                ins = shape_dbinfos[ins_idx]                    
                if ins["difficulty"] == 0:
                    if ins["score"] > score_threshold:
                        if ins["geo_conf"] > geo_conf_threshold:
                            sub_db_infos[class_name][img_shape].append(ins)
                            geo_count = geo_count + 1
            total_valid_ins += geo_count
            total_ins += len(shape_dbinfos)
            print(class_name, img_shape, "geo_count: ", geo_count, "total", len(shape_dbinfos))

    with open(new_db_infos, 'wb') as f:
        pickle.dump(sub_db_infos, f)
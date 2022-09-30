python uncertainty_estimator.py --kitti_root ../datasets/kitti --pred_folders <path-to-pred_folders>/
python create_data.py --kitti_root ../datasets/kitti --ssl True
python create_background_infos.py --kitti_root ../datasets/kitti
python parse_db_infos.py --old_db_infos ../datasets/kitti/kitti_dbinfos_test.pkl --new_db_infos ../datasets/kitti/kitti_dbinfos_test_filtered.pkl --score_threshold 0.7 --geo_conf_threshold 0.75
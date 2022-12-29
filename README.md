# Mix-Teaching: A General Semi-Supervised Learning Framework for Monocular 3D Object Detection

<img src="figures/Mix-Teaching.jpg" alt="vis2" style="zoom:30%" />

This is the official implementation of our manuscript [**Mix-Teaching: a general semi-supervised learning framework for monocular 3D object detection**](https://arxiv.org/abs/2207.04448v1). The raw data of KITTI which consists of 48K temporal images is used as unlabeled data in all experiments. For more details, please see our paper.

The performance on KITTI validation set (3D) is as follows:
<table align="center">
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">10%</td>    
        <td colspan="3",div align="center">30%</td>  
        <td colspan="3",div align="center">100%</td>  
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td> 
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td> 
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td>  
    </tr>
    <tr>
        <td div align="center">MonoFlex</td>
        <td div align="center">5.76</td> 
        <td div align="center">4.67</td> 
        <td div align="center">3.54</td> 
        <td div align="center">15.58</td> 
        <td div align="center">11.03</td> 
        <td div align="center">8.93</td> 
        <td div align="center">23.64</td> 
        <td div align="center">17.51</td> 
        <td div align="center">14.83</td>  
    </tr>    
    <tr>
        <td div align="center">Ours</td>
        <td div align="center">14.43</td> 
        <td div align="center">10.65</td> 
        <td div align="center">8.41</td> 
        <td div align="center">23.81</td> 
        <td div align="center">16.94</td> 
        <td div align="center">13.80</td> 
        <td div align="center">30.82</td> 
        <td div align="center">22.18</td> 
        <td div align="center">18.61</td>  
    </tr>
    <tr>
        <td div align="center">Abs. Imp.</td>
        <td div align="center">+8.57</td> 
        <td div align="center">+5.98</td> 
        <td div align="center">+4.87</td> 
        <td div align="center">+8.23</td> 
        <td div align="center">+5.91</td> 
        <td div align="center">+4.87</td> 
        <td div align="center">+7.18</td> 
        <td div align="center">+4.67</td> 
        <td div align="center">+3.78</td>  
    </tr>
</table>


## Getting Started
### 1. Installation
Please refer to [Installation](MonoFlex/README.md)

Then run
```console
pip install mmcv-full==1.2.5 mmdet==2.11.0
git clone https://github.com/open-mmlab/mmdetection3d && cd mmdetection3d && git checkout v0.9.0
cd ../ && pip install mmdetection3d/
```

### 2. Dataset
Please first download the training set and organize it as following structure:

```
datasets
│──kitti
│    ├──ImageSets
│    ├──training        <-- 7481 train data
│    │   ├──calib 
│    │   ├──label_2 
│    │   └──image_2
│    └──testing         <-- empty directory to save raw data in official format
│        ├──calib 
│        ├──image_2
│        └──ImageSets
└──raw_data             <-- raw data in zip format
```  

Download and transfer format for KITTI raw data.

```console
cd datasets && mkdir raw_data
cd ../raw_data_tools && bash download_raw_data.sh ../datasets/raw_data
python convert_det_format.py --raw_data_root ../datasets/raw_data --kitti_root ../datasets/kitti
cd ../pseudo_labeling_tools && python generate_imageset.py --kitti_root ../datasets/kitti
```
Then run
```console
python create_data.py --kitti_root ../datasets/kitti
```

### 3. Train teacher model
Please refer to [Training](MonoFlex/README.md) in supervised mode.

### 4. Generate pseudo labels for unlabeled data
Please refer to [Inference](MonoFlex/README.md).

Inference on unlabeled data and organize results as following structure:
```
pred_folders
│──model_1_preds
│    ├──000000.txt
│    ├──000001.txt
│    └── ...
│──model_2_preds
│    ├──000000.txt
│    ├──000001.txt
│    └── ...
└── ...            
```  

### 5. Pseudo labeling
```console
python uncertainty_estimator.py --kitti_root ../datasets/kitti --pred_folders <path-to-pred_folders>/
python create_data.py --kitti_root ../datasets/kitti --ssl True
python create_background_infos.py --kitti_root ../datasets/kitti
python parse_db_infos.py --old_db_infos ../datasets/kitti/kitti_dbinfos_test.pkl --new_db_infos ../datasets/kitti/kitti_dbinfos_test_filtered.pkl --score_threshold 0.7 --geo_conf_threshold 0.75
```
or
```console
bash pseudo_labeling.sh
```
### 6. Train student model with labeled and unlabeled data
Please refer to [Training](MonoFlex/README.md) in semi-supervised model.

### 7. Continue with step 4.


## Citation
If you find our work useful in your research, please consider citing:

```latex
@article{Yang2022MixTeachingAS,
  title={Mix-Teaching: A Simple, Unified and Effective Semi-Supervised Learning Framework for Monocular 3D Object Detection},
  author={Lei Yang and Xinyu Zhang and Li Wang and Minghan Zhu and Chuan-Fang Zhang and Jun Li},
  journal={ArXiv},
  year={2022},
  volume={abs/2207.04448}
}
```

## Acknowledgements
Thank for the excellent cooperative perception codebases [MonoFlex](https://arxiv.org/pdf/2104.02323.pdf)

Thank for the excellent perception datasets [KITTI](https://www.cvlibs.net/datasets/kitti/)

## Contact

If you have any problem with this code, please feel free to contact **yanglei20@mails.tsinghua.edu.cn**.
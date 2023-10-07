# Memory-and-Anticipation Transformer for Online Action Understanding

## Introduction

This is a PyTorch implementation for our ICCV 2023 paper "[`Memory-and-Anticipation Transformer for Online Action Understanding`](https://echo0125.github.io/mat/)".

![network](demo/Framework.png?raw=true)

## Environment

- The code is developed with CUDA 10.2, ***Python >= 3.7.7***, ***PyTorch >= 1.7.1***

    0. [Optional but recommended] create a new conda environment.
        ```
        conda create -n mat python=3.7.7
        ```
        And activate the environment.
        ```
        conda activate mat
        ```

    1. Install the requirements
        ```
        pip install -r requirements.txt
        ```

## Data Preparation

### Pre-extracted Feature

You can directly download the pre-extracted feature (.zip) from the UTBox links provided by [`TeSTra`](https://github.com/zhaoyue-zephyrus/TeSTra#pre-extracted-feature).


### (Alternative) Prepare dataset from scratch

You can also try to prepare the datasets from scratch by yourself. 

#### THUMOS14 and TVSeries

For THUMOS14 and TVSeries, please refer to [`LSTR`](https://github.com/amazon-research/long-short-term-transformer#data-preparation).

#### EK100

For EK100, please find more details at [`RULSTM`](https://github.com/fpv-iplab/rulstm).

### Data Structure

1. If you want to use our [dataloaders](src/rekognition_online_action_detection/datasets), please make sure to put the files as the following structure:

   * THUMOS'14 dataset:
       ```
       $YOUR_PATH_TO_THUMOS_DATASET
       ├── rgb_kinetics_resnet50/
       |   ├── video_validation_0000051.npy (of size L x 2048)
       │   ├── ...
       ├── flow_kinetics_bninception/
       |   ├── video_validation_0000051.npy (of size L x 1024)
       |   ├── ...
       ├── target_perframe/
       |   ├── video_validation_0000051.npy (of size L x 22)
       |   ├── ...
       ```
   
   
   * TVSeries dataset:
       ```
          $YOUR_PATH_TO_TVSERIES_DATASET
          ├── rgb_kinetics_resnet50/
          |   ├── Breaking_Bad_ep1.npy (of size L x 2048)
          │   ├── ...
          ├── flow_kinetics_bninception/
          |   ├── Breaking_Bad_ep1.npy (of size L x 1024)
          |   ├── ...
          ├── target_perframe/
          |   ├── Breaking_Bad_ep1.npy (of size L x 31)
          |   ├── ...
       ```
   
   
   
   * EK100 dataset:
       ```
          $YOUR_PATH_TO_EK_DATASET
          ├── rgb_kinetics_bninception/
          |   ├── P01_01.npy (of size L x 1024)
          │   ├── ...
          ├── flow_kinetics_bninception/
          |   ├── P01_01.npy (of size L x 1024)
          |   ├── ...
          ├── target_perframe/
          |   ├── P01_01.npy (of size L x 3807)
          |   ├── ...
          ├── noun_perframe/
          |   ├── P01_01.npy (of size L x 301)
          |   ├── ...
          ├── verb_perframe/
          |   ├── P01_01.npy (of size L x 98)
          |   ├── ...
       ```
   
2. Create softlinks of datasets:

    ```
    cd memory-and-anticipation-transformer
    ln -s $YOUR_PATH_TO_THUMOS_DATASET data/THUMOS
    ln -s $YOUR_PATH_TO_TVSERIES_DATASET data/TVSeries
    ln -s $YOUR_PATH_TO_EK_DATASET data/EK100
    ```

## Training

The commands are as follows.

```
cd memory-and-anticipation-transformer
# Training from scratch
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES
# Finetuning from a pretrained model
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
    MODEL.CHECKPOINT $PATH_TO_CHECKPOINT
```

## Online Inference

There are *two kinds* of evaluation methods in our code.

* First, you can use the config `SOLVER.PHASES "['train', 'test']"` during training. This process devides each test video into non-overlapping samples, and makes prediction on the all the frames in the short-term memory as if they were the latest frame. Note that this evaluation result is ***not*** the final performance, since (1) for most of the frames, their short-term memory is not fully utlized and (2) for simplicity, samples in the boundaries are mostly ignored.

    ```
    cd memory-and-anticipation-transformer
    # Inference along with training
    python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        SOLVER.PHASES "['train', 'test']"
    ```

* Second, you could run the online inference in `batch mode`. This process evaluates all video frames by considering each of them as the latest frame and filling the long- and short-term memories by tracing back in time. Note that this evaluation result matches the numbers reported in the paper. On the other hand, this mode can run faster when you use a large batch size, and we recomand to use it for performance benchmarking.

    ```
    cd memory-and-anticipation-transformer
    # Online inference in batch mode
    python tools/test_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.LSTR.INFERENCE_MODE batch
    ```
    
## Main Results and checkpoints

### THUMOS14

|       method      | feature   |  mAP (%)  |                             config                                                |   checkpoint   |
|  :--------------: |  :-------------:  |  :-----:  |  :-----------------------------------------------------------------------------:  |  :----------:  |
|  MAT           |  Anet v1.3 |   70.5    | [yaml](configs/THUMOS/MAT/mat_long_256_work_8_anet_1x.yaml) | [Download](https://drive.google.com/file/d/1NyhGSSIBd_T9osbnc2_S2OZG4HoMcvoX/view?usp=drive_link) |
|  MAT           |    Kinetics    |   71.6    | [yaml](configs/THUMOS/MAT/mat_long_256_work_8_kinetics_1x.yaml)      | [Download](https://drive.google.com/file/d/1W3OlCOz4rkRq4MV_RvO8mHLzmdZrrnbF/view?usp=drive_link) |

### EK100

|  method  |    feature    |  verb (overall)  |  noun (overall)  |  action (overall)  |  config  |                                checkpoint                                |
|  :----:  |  :-------------:  |  :------------:  |  :------------:  |  :--------------:  |  :----:  |  :--------------------------------------------------------------------:  |
|  MAT  |  RGB+FLOW  |      35.0      |       38.8       |        19.5        |  [yaml](configs/EK100/MAT/mat_long_64_work_5_kinetics_1x.yaml) | [Download](https://drive.google.com/file/d/1qVz1EuIZ7pUKRjn2Udq6h82xwDrHDePY/view?usp=drive_link) |

## Citations

If you are using the data/code/model provided here in a publication, please cite our paper:
```
@inproceedings{wang2023memory,
               title={Memory-and-Anticipation Transformer for Online Action Understanding},
               author={Wang, Jiahao and Chen, Guo and Huang, Yifei and Wang, Limin and Lu, Tong},
               booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
               pages={13824--13835},
               year={2023}
}
```
## License

This project is licensed under the Apache-2.0 License.

## Acknowledgements

This codebase is built upon [`LSTR`](https://github.com/amazon-research/long-short-term-transformer).

The code snippet for evaluation on EK100 is borrowed from [`TeSTra`](https://github.com/zhaoyue-zephyrus/TeSTra).

Also, thanks to Mingze Xu and Yue Zhao for assistance to reproduce the feature.

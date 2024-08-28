# LDCTIQAC2023

This repository contains the source code for the models submitted to the [Low-dose Computed Tomography Perceptual Image Quality Assessment Challenge 2023](). Click below to get the code for each method:

> [**agaldran**](https://github.com/agaldran/ldct_iqa)
> 
> **RPI_AXIS**
>
> **CHILL@UK**
>
> **FeatureNet**
>
> **Team Epoch**
>
> **gabybaldeon**

## Usage

The code in this repository is for inference. To test the models locally, edit the paths for the input file, the output file, and the model weights in `process.py`, and then run `process.py`. `process.py` expects a 3D volume as input. When a 3D volume is provided, assessments of each slice in the volume will be conducted. 

For training, implement the model code into the user's project. The dataset for training the models can be downloaded from [Zenodo](https://zenodo.org/records/7833096#.ZEFywOxBzn5).

The model weights can be downloaded from this [link](https://drive.google.com/drive/folders/1okL-YTiFlwogmF_tvYjCxGsUKV2DJ2u0?usp=sharing).

## Reference Docker

These codes are built based on the reference docker, which is for submitting to the [challenge website](https://ldctiqac2023.grand-challenge.org/). Refer to this [repository](https://github.com/Ewha-AI/LDCTIQAC_reference_docker) for the reference docker.

## Citation

    @article{}

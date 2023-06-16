#!/bin/bash
# This script extracts the keypoints and descriptors from a dataset of images.
# Make sure to pass the correct DATSET_BASE_PATH.
#
# For each image it will describe the image with and without flipping
# and save those descriptors in a .npy file.

DESCRIPTOR="vlfeat_sift_heq"
DATASET_BASE_PATH="descriptors/spp-jsons/"
#DATASET_BASE_PATH="descriptors/spp-ext-v1-jsons/"
#DATASET_BASE_PATH="descriptors/spp-ext-v2-jsons/"

for type in Microscopy FlowCytometry BodyImaging Blots; do
    echo "Creating evidence database for $type"

    FLIP=false
    CLASS_NAME=$type
    DATASET=$DATASET_BASE_PATH$type"-dataset.json"
    
    echo "Describe image without flipping"
    python run_dataset_descriptor.py \
        --dataset $DATASET \
        --descriptor $DESCRIPTOR\
        --flip $FLIP \
        --class-name $CLASS_NAME

    FLIP=true
    echo "Describe image with flipping"
    python run_dataset_descriptor.py \
        --dataset $DATASET \
        --descriptor $DESCRIPTOR\
        --flip $FLIP \
        --class-name $CLASS_NAME
done
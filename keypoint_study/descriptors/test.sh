#!/bin/bash
# This script extract the keypoints and descriptors from a dataset of images
# Make sure to have the dataset in the same folder as this script

DESCRIPTOR="vlfeat_sift_heq" # Descriptor we use to achieve our results from UPM
#DESCRIPTOR="vlfeat_sift" # Descriptor we use to achieve our results from UPM
BASE_PATH="descriptors/spm-jsons/"

echo "Creating evidence database for $type"

FLIP=false
CLASS_NAME="Blots"
DATASET=$BASE_PATH$CLASS_NAME"-dataset.json"
    
echo "Describe image without flipping"
python descriptors/run_dataset_descriptor.py \
    --dataset $DATASET \
    --descriptor $DESCRIPTOR\
    --flip $FLIP \
    --class-name $CLASS_NAME

FLIP=true
echo "Describe image with flipping"
python descriptors/run_dataset_descriptor.py \
    --dataset $DATASET \
    --descriptor $DESCRIPTOR\
    --flip $FLIP \
    --class-name $CLASS_NAME


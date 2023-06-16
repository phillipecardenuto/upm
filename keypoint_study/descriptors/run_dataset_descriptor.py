#from image_descriptor import sila_descriptor
from image_descriptor import cv_sift
from image_descriptor import vlfeat_phow
from image_descriptor import vlfeat_dsift
from image_descriptor import vlfeat_sift

from tqdm.contrib.concurrent import process_map

import json
from pathlib import Path
import os
import numpy as np
from glob import glob


import argparse

def file_output_exists(panel_id, descriptor):
    global flip
    global class_name


    panel_output =  f"description-{class_name}/{panel_id}"
    
    if os.path.isdir(panel_output):
        panel_output_kps = f"{panel_output}/{descriptor}_flip_kps.npy" if flip else \
                f"{panel_output}/{descriptor}_kps.npy"  
        panel_output_desc = f"{panel_output}/{descriptor}_flip_desc.npy" if flip else \
                f"{panel_output}/{descriptor}_desc.npy"  

        if os.path.isfile(panel_output_kps) and os.path.isfile(panel_output_desc):
            return True
    return False


def describe_panel(input):
    global descriptor
    global flip
    panel_id, panel = input

    if file_output_exists(panel_id, descriptor):
        return
    
    if descriptor == 'sila':
        raise NotImplementedError
        #kpts, descs = sila_method(panel, flip=flip) 
    elif descriptor == 'cv_rsift_heq':
        kpts, descs = cv_sift(panel, hist_equalization=True, root_sift=True, flip=flip)
    elif descriptor == 'cv_rsift':
        kpts, descs = cv_sift(panel, hist_equalization=False, root_sift=True, flip=flip)
    elif descriptor == 'cv_sift_heq':
        kpts, descs = cv_sift(panel, hist_equalization=True, root_sift=False, flip=flip)
    elif descriptor == 'cv_sift':
        kpts, descs = cv_sift(panel, hist_equalization=False, root_sift=False, flip=flip)
    elif descriptor == 'vlfeat_phow':
        kpts, descs = vlfeat_phow(panel, flip=flip)
    elif descriptor == 'vlfeat_dsift':
        kpts, descs = vlfeat_dsift(panel, flip=flip)
    elif descriptor == 'vlfeat_rsift_heq':
        kpts, descs = vlfeat_sift(panel, hist_equalization=True, root_sift=True, flip=flip)
    elif descriptor == 'vlfeat_rsift':
        kpts, descs = vlfeat_sift(panel, hist_equalization=False, root_sift=True, flip=flip)
    elif descriptor == 'vlfeat_sift_heq':
        kpts, descs = vlfeat_sift(panel, hist_equalization=True, root_sift=False, flip=flip)
    elif descriptor == 'vlfeat_sift':
        kpts, descs = vlfeat_sift(panel, hist_equalization=False, root_sift=False, flip=flip)
    
    panel_output =  f"description-{class_name}/{panel_id}"
    os.makedirs(panel_output, exist_ok=True)

    kps_name = f"{panel_output}/{descriptor}_flip_kps.npy" if flip else \
                f"{panel_output}/{descriptor}_kps.npy"  

    desc_name = f"{panel_output}/{descriptor}_flip_desc.npy" if flip else \
                f"{panel_output}/{descriptor}_desc.npy"  
    np.save(kps_name, kpts)
    np.save(desc_name, descs)

    return

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(dataset, class_name, n_jobs=-1):

    output = f"description-{class_name}"

    os.makedirs(output, exist_ok=True)

    #panels = glob(f"{dataset}/*/*.png")
    #print(f"Found {len(panels)} panels.")
    
    with open(dataset) as f:
        data = json.load(f)
        panels = [(k, d['panel_path']) for k, d in data.items()]

    process_map(describe_panel, panels, chunksize=1, max_workers=n_jobs)

def str2bool(v):
    """Convert string to boolean.
    REF: https://stackoverflow.com/a/43357954
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset.')
    parser.add_argument('--descriptor', type=str, required=True, help='Descriptor name.')
    parser.add_argument('--flip', type=str2bool, const=True, default=True, nargs='?',
                         help='Flip image before description.')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs.')
    parser.add_argument('--class-name', type=str, required=True, help='Path to dataset.')
    args = parser.parse_args()

    global descriptor
    global flip
    global class_name
    descriptor = args.descriptor
    flip = args.flip

    class_name = args.class_name
    dataset = args.dataset
    n_jobs = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() + 4)

    main(dataset, class_name, n_jobs)
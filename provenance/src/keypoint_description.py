from src.image_descriptor import sila_method, cv_sift, vlfeat_phow, vlfeat_dsift, vlfeat_sift
from tqdm.contrib.concurrent import process_map
import json
import os
import numpy as np
import argparse

def parse_opt():
    """
    Setup the ablation parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset.')
    parser.add_argument('--descriptor-type', default='vlfeat_sift_heq', type=str, help='Image Decriptor type (e.g., vlfeat_sift_heq')
    opt = parser.parse_args()
    return opt


def desc_exists(panel_id, method):

    panel_output =  f"description/{panel_id}"
    if os.path.isdir(panel_output):
        panel_output_kps = panel_output +f"/{method}_kps.npy"
        panel_output_desc = panel_output +f'/{method}_desc.npy'
        if os.path.isfile(panel_output_kps) and os.path.isfile(panel_output_desc):
            return True
    return False

def run_descriptor(panel, method, flip=False):
    if method == 'sila':
        kpts, descs = sila_method(panel, flip=flip)
    elif method == 'cv_rsift_heq':
        kpts, descs = cv_sift(panel, hist_equalization=True, root_sift=True, flip=flip)
    elif method == 'cv_rsift':
        kpts, descs = cv_sift(panel, hist_equalization=False, root_sift=True, flip=flip)
    elif method == 'cv_sift_heq':
        kpts, descs = cv_sift(panel, hist_equalization=True, root_sift=False, flip=flip)
    elif method == 'cv_sift':
        kpts, descs = cv_sift(panel, hist_equalization=False, root_sift=False, flip=flip)
    elif method == 'vlfeat_phow':
        kpts, descs = vlfeat_phow(panel, flip=flip)
    elif method == 'vlfeat_dsift':
        kpts, descs = vlfeat_dsift(panel, flip=flip)
    elif method == 'vlfeat_rsift_heq':
        kpts, descs = vlfeat_sift(panel, hist_equalization=True, root_sift=True, flip=flip)
    elif method == 'vlfeat_rsift':
        kpts, descs = vlfeat_sift(panel, hist_equalization=False, root_sift=True, flip=flip)
    elif method == 'vlfeat_sift_heq':
        kpts, descs = vlfeat_sift(panel, hist_equalization=True, root_sift=False, flip=flip)
    elif method == 'vlfeat_sift':
        kpts, descs = vlfeat_sift(panel, hist_equalization=False, root_sift=False, flip=flip)
    return kpts, descs



def describe_panel(input):
    panel_id, panel, method = input

    panel_output =  f"description/{panel_id}"
    # Without flip
    if not desc_exists(panel, method):
        kpts, descs = run_descriptor(panel, method, flip=False)
        os.makedirs(panel_output, exist_ok=True)
        np.save(panel_output+f"/{method}_kps.npy", kpts)
        np.save(panel_output+f"/{method}_desc.npy", descs)
    # With flip
    if not desc_exists(panel, method+'_flip'):
        kpts, descs = run_descriptor(panel, method, flip=True)
        os.makedirs(panel_output, exist_ok=True)
        np.save(panel_output+f"/{method}_flip_kps.npy", kpts)
        np.save(panel_output+f"/{method}_flip_desc.npy", descs)
    return


if __name__ == "__main__":
    opt = parse_opt()

    with open(opt.dataset) as f:
        dataset = json.load(f)

    panels = [(k, d['panel_path'], 'vlfeat_sift_heq') for k, d in dataset.items()]
    process_map(describe_panel, panels, chunksize=1)
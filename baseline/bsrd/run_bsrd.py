import cv2
from glob import glob
import shutil
from bioScale import *
from tqdm.contrib.concurrent import thread_map
import argparse
from itertools import combinations
import numpy as np
import json

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--papermill-path', type=str, required=True, help='Path to stock-papermill')
    opt = parser.parse_args()
    return opt


def run_cmfd(cmfd_input):
    
    pi, pj = cmfd_input
    if pi == pj:
        adjacency_matrix[pi][pj] = 0 
        return
    
    if adjacency_matrix[pi][pj] >= 0:
        return
        
    panel_i = cv2.imread(papermill_panels[pi])
    panel_j = cv2.imread(papermill_panels[pj])
    
    # Describe Images
    kpi, desc_i = sift_detect_sift_describe(panel_i)
    kpj, desc_j = sift_detect_sift_describe(panel_j)
    
    matches = match_keypoints(panel_i.shape, kpi, desc_i, panel_j.shape, kpj, desc_j)
    
    
    adjacency_matrix[pi][pj] = len(matches)
    adjacency_matrix[pj][pi] = len(matches)
    
    
    
args = parse_opt()
print(args)
papermill_path = args.papermill_path
with open(papermill_path) as p:
    papermill_panels = json.load(p)
papermill_panels = [p['panel_path'] for p in papermill_panels.values()]
adjacency_matrix = - np.ones((len(papermill_panels), len(papermill_panels)))

cmfd_input = list(combinations(range(len(papermill_panels)),2))

thread_map(run_cmfd, cmfd_input, max_workers=70, desc=f'Bio-Scale')

print(adjacency_matrix)

np.save("adjacency_matrix",adjacency_matrix)
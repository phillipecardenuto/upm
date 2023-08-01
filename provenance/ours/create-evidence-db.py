from glob import glob
import sys
from pathlib import Path
import pickle
from forensic_lib.scientificEvidences.imgEv.CNN import *
from forensic_lib.visualization.image_match import *
from forensic_lib.utils.img_utils import load_image
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os

CLASSES=['Blots', 'BodyImaging', 'Microscopy', 'FlowCytometry', 'Graphs']
DATASETS=['spm', 'spm-v1', 'spm-v2']

model, preprocess = load_default_image_model_or_preprocess()

for db in (pbar := tqdm(DATASETS)):
    for cl in CLASSES:
        pbar.set_description(f"Processing {db} - Class={cl}\n")

        dataset_path = f'{db}/{cl}-dataset.json'
        with open(dataset_path) as p:
            panels_dataset = json.load(p)
            panels = [p['panel_path'] for p in panels_dataset.values()]

            panels_ids = [i+1 for i in range(len(panels))]
            image_embeddings , embeddings_id = get_image_embedding(panels, panels_ids,model,
                                                                   preprocess, normalize=True,
                                                                   use_gpu= True,  gpu_id=0)

            os.makedirs(f'EV-DB/{db}/',exist_ok=True)

        for p_id, p in enumerate(panels):
            p = Path(p)
            panels_dataset[str(p_id)]["panels_image_embedding"] = image_embeddings[p_id]

        with open(f'EV-DB/{db}/{cl}-evidence-db.json','w') as f:
           json.dump(panels_dataset, f)

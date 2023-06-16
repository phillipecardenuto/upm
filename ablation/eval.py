"""Ablation study evaluation.

To use this script, make sure to download the folowing files:
- ablation-extended-db.json (https://drive.google.com/file/d/1F8m5zMHGqIIEfIAC9VhIUdapuHt2DqQ-/view)
- ablation-results.zip (https://drive.google.com/file/d/1G1q2McF6E6-yaNk0ag8B_yqOl3D1zel1/view)

This script will generate the following files:
- panelEvaluation.csv: panel-level evaluation results
- panelGroupEvaluation.csv: panel-level evaluation results per class
- panelRelationshipEval.csv: panel-level evaluation results per class
- docEvaluation.csv: document-level evaluation results

Author: João Phillipe Cardenuto
"""
from glob import glob
import sys
from generate_groups import provenance_groups
from panel2doc import panel2doc_provenance
import json
from metrics import panel_evaluation, doc_evaluation
import pandas as pd
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import numpy as np
from datetime import datetime

with open('ablation-extended-db.json') as f:
    panels_dataset = json.load(f)
    
with open('annotation.json') as f:
    our_annotation = json.load(f)
    
with open('document-level-annotation.json') as f:
    doc_annotation = json.load(f)
with open('extended-document-stock-photo-dataset.json') as f:
    doc_dataset = json.load(f)

def parse_parameters(res):
    
    resname = Path(res).stem.split('-')
    
    results = {}
    results['desc'] = resname[0]
    results['matching'] = resname[1]
    results['alignment'] = resname[2]
    results['rank_top_k'] = resname[3]
    results['min_keypoints'] = resname[4]
    results['max_queuesize'] = resname[5]
    results['min_area'] = resname[6]
    results['same_class'] = resname[7]
    
    return results
    
def digest_panel_results(res):
    
    df = pd.DataFrame()
    df_group_per_class = pd.DataFrame()
    df_relationship_per_class = pd.DataFrame()
    
    parameters = parse_parameters(res)
    
    predicted_adjacency_matrix = np.load(res)
    predicted_graphs = provenance_groups(panels_dataset, predicted_adjacency_matrix)
    metrics = panel_evaluation(predicted_graphs, our_annotation, panels_dataset, predicted_adjacency_matrix, None)
    
    r = {}
    grouping_quality_per_class = {}
    relationship_quality_per_class = {}

    for k,item in metrics.items():

        if 'grouping_quality_per_class' in k:
            for cl, val in item.items():
                grouping_quality_per_class[cl] = val
        elif 'relationship_quality_per_class' in k:
            for cl, val in item.items():
                relationship_quality_per_class[cl] = val
        else:
            r[k] = item
    r.update(parameters)
    grouping_quality_per_class.update(parameters)
    relationship_quality_per_class.update(parameters)
    
    df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
    df_group_per_class = pd.concat([df_group_per_class, pd.DataFrame([grouping_quality_per_class])], ignore_index=True)
    df_relationship_per_class = pd.concat([df_relationship_per_class, pd.DataFrame([relationship_quality_per_class])], ignore_index=True)
    
    return df, df_group_per_class, df_relationship_per_class
    
def digest_doc_results(res):
    
    df = pd.DataFrame()
    parameters = parse_parameters(res)
    
    predicted_adjacency_matrix = np.load(res)
    predicted_graphs = provenance_groups(panels_dataset, predicted_adjacency_matrix)
    predicted_doc_graphs,predicted_adjacency_matrix  = panel2doc_provenance(predicted_graphs,predicted_adjacency_matrix, panels_dataset,doc_dataset)

    metrics = doc_evaluation(predicted_doc_graphs, doc_annotation, doc_dataset, predicted_adjacency_matrix, None)
    
    r = metrics
    r.update(parameters)
    
    df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
    
    return df

print("Evaluating Document level results")
results = glob("results/*.npy")
list_dfs = process_map(digest_doc_results, results, max_workers=10)
document_eval = pd.concat(list_dfs, ignore_index=True)
document_eval.to_csv("docEvaluation"+str(datetime.now()).split('.')[0].replace(' ','_').replace(':','-')+".csv")

print("Evaluating Panel level results")
results = glob("results/*.npy")
df, df_group_class, df_rel_class = digest_panel_results(results[0])
for res in tqdm(results[1:], total=len(results)-1, desc="Panel level Evaluation"):
    d, dg, dr = digest_panel_results(res)
    df = pd.concat([df, d] , ignore_index=True)
    df_group_class = pd.concat([df_group_class, dg], ignore_index=True)
    df_rel_class = pd.concat([df_rel_class, dr], ignore_index=True)

print("Saving panel evaluation results")
df.to_csv('panelEvaluation' +str(datetime.now()).split('.')[0].replace(' ','_').replace(':','-')+".csv")
df_group_class.to_csv('panelGroupEvaluation' +str(datetime.now()).split('.')[0].replace(' ','_').replace(':','-')+".csv")
df_rel_class.to_csv('panelRelationshipEval' +str(datetime.now()).split('.')[0].replace(' ','_').replace(':','-')+".csv")
print("Done!")
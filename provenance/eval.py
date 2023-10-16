from glob import glob
from src.generate_groups import panel_provenance_groups, document_provenance
import json
from src.metrics import panel_evaluation, doc_evaluation
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime

with open('dataset/dev_dataset.json') as f:
    panels_dataset = json.load(f)

with open('dataset/dev_dataset_gt.json') as f:
    our_annotation = json.load(f)

with open('dataset/dev_doc_annotation.json') as f:
    doc_annotation = json.load(f)
with open('dataset/dev_doc.json') as f:
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
    predicted_graphs = panel_provenance_groups(panels_dataset, predicted_adjacency_matrix)
    metrics = panel_evaluation(predicted_graphs, our_annotation, panels_dataset, predicted_adjacency_matrix)

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
    df = df.rename(columns={'relationship_precision':'pairing_precision',
                            'relationship_recall': 'pairing_recall',
                            'relationship_quality': 'content_pairing',
                            'grouping_quality': 'content_grouping', 
                            'global_precision': "classification_precision",
                            'global_recall': "classification_recall",
                            'global_quality': "content_classification",
                    })
    return df, df_group_per_class, df_relationship_per_class

def digest_doc_results(res):

    df = pd.DataFrame()
    parameters = parse_parameters(res)
    predicted_adjacency_matrix = np.load(res)
    predicted_doc_graphs, predicted_adjacency_matrix  = document_provenance(panels_dataset, predicted_adjacency_matrix)
    metrics = doc_evaluation(predicted_doc_graphs, doc_annotation, doc_dataset, predicted_adjacency_matrix)

    r = metrics
    r.update(parameters)

    df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
    df = df.rename(columns={'relationship_precision':'pairing_precision',
                            'relationship_recall': 'pairing_recall',
                            'relationship_quality': 'content_pairing',
                            'grouping_quality': 'content_grouping', 
                            'global_precision': "classification_precision",
                            'global_recall': "classification_recall",
                            'global_quality': "content_classification",
                    })

    return df

print("Evaluating Document level results")
results = glob("results/*.npy") # change this line to the resultant provenance matrix
document_eval = digest_doc_results(results[0])
document_eval.to_csv("docEvaluation"+str(datetime.now()).split('.')[0].replace(' ','_').replace(':','-')+".csv")

print("Evaluating Panel level results")
results = glob("results/*.npy")  # change this line to the resultant provenance matrix
df, df_group_class, df_rel_class = digest_panel_results(results[0])

print("Saving panel evaluation results")
df.to_csv('panelEvaluation' +str(datetime.now()).split('.')[0].replace(' ','_').replace(':','-')+".csv")
df_group_class.to_csv('panelGroupEvaluation' +str(datetime.now()).split('.')[0].replace(' ','_').replace(':','-')+".csv")
df_rel_class.to_csv('panelPairingEval' +str(datetime.now()).split('.')[0].replace(' ','_').replace(':','-')+".csv")
print("Done!")

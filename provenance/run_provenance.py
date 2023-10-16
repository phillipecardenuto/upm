import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.match import homography_alignment_area, cluster_alignment_area
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
import json
import argparse
import os
from pathlib import Path
from src.utils import validate_args
from src.keypoint_description import describe_panel
from src.generate_groups import panel_provenance_groups, document_provenance

def parse_opt():
    """
    Setup the ablation parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset.')
    parser.add_argument('--descriptor-type', default='vlfeat_sift_heq', type=str, help='Image Decriptor type (e.g., cv_rsift')
    parser.add_argument('--matching-strategy', type=str, default='BF', help='Matching Strategy Type (e.g., BF)')
    parser.add_argument('--alignment-strategy', type=str, default='CV_MAGSAC', help='Keypoints Alignment Method.')
    parser.add_argument('--top-k', type=int, default=400, help='Number of retrieved results from the filtering step.')
    parser.add_argument('--min-keypoints', type=int, default=20, help='Minimun number of keypoints matching.')
    parser.add_argument('--max-queuesize', type=int, default=300, help='Maximum number of processing items per iteraction.')
    parser.add_argument('--use-area', type=bool, default=False, required=False, help='If True, the content shared area of the images will be used in the provenance analysis.')
    parser.add_argument('--min-area', type=float, default=0.01, help='Minimum shared area to consider two images suspect of sharing content')
    parser.add_argument('--same-class', type=bool, default=True, help='Perform provenance using images from the same classes')
    parser.add_argument('--visualize', type=bool, default=False, help='Visualize the provenance graph')
    opt = parser.parse_args()
    
    return opt

def get_image_shared_area(query, qs):
    """
    Returns the shared area between two images
    """

    global matching_strategy
    global alignment_strategy
    global descriptor_type
    global min_keypoints

    # Read infos
    qs_info = get_panel_info_by_id(qs)
    query_info = get_panel_info_by_id(query)
    query_flip_info = get_panel_info_by_id(query,flip=True)

    try:
        if alignment_strategy == 'cluster':
            rg_shared_area =  cluster_alignment_area(query_info, qs_info, min_match_count=min_keypoints, matching_method=matching_strategy)
            flip_shared_area=  cluster_alignment_area(query_flip_info, qs_info, min_match_count=min_keypoints)
        else:
            rg_shared_area =  homography_alignment_area(query_info, qs_info, alignment_method=alignment_strategy, matching_method=matching_strategy, min_kpts_matches=min_keypoints)
            flip_shared_area=  homography_alignment_area(query_flip_info, qs_info, alignment_method=alignment_strategy, matching_method=matching_strategy, min_kpts_matches=min_keypoints)

        # If some content match, check shared_content_area
        shared_area = max(max(rg_shared_area), max(flip_shared_area))
        if shared_area > min_area:
            if shared_area in rg_shared_area:
                return rg_shared_area 
            return flip_shared_area
    except Exception as e:
        print(e)
        return (0,0)
    return (0,0)

def retrieve_topk_similar(similarity_matrix, query, k=100, same_class=False):
    """
    Retrieves all similar entities to the query that do not
    belong to the same document as the query
    We had to remove the entities from the same document, since we want to
    spot entities across documents (i.e., paper mills).
    """
    query_doc = panels_dataset[str(query)]['doc_id']
    query_class = panels_dataset[str(query)]['panel_class']
    # Get The top plus safe margin
    top_similar = np.argsort(similarity_matrix[query],)[::-1][:2*k]

    # Remove entities that share the same figure as the query
    remove_list = []
    for i in top_similar:
        if same_class:
            # Remove panels from a different class of the query
            if query_class != panels_dataset[str(i)]['panel_class']:
                remove_list.append(i)

        # Remove panels that belong to the same document as the query
        if query_doc == panels_dataset[str(i)]['doc_id']:
            remove_list.append(i)

    # Remove entities
    for i in remove_list:
        top_similar = top_similar[top_similar != i]

    # Assert top_similar is at most k sized
    top_similar = top_similar[:k]
    return list(top_similar)

def get_panel_info_by_id(_id, flip=False):
    global panels_dataset
    global descriptor_type

    
    panel_info = {}
    panel_info['image_path'] = panels_dataset[str(_id)]['panel_path']
    if flip:
        panel_info['keypoints_path'] = f"description/{_id}/{descriptor_type}_flip_kps.npy"
        panel_info['desc_path'] = f"description/{_id}/{descriptor_type}_flip_desc.npy"
    else:
        panel_info['keypoints_path'] = f"description/{_id}/{descriptor_type}_kps.npy"
        panel_info['desc_path'] = f"description/{_id}/{descriptor_type}_desc.npy"

    return panel_info

def max_shared_area(shared_area1,shared_area2):
    if max(shared_area1) > max(shared_area2):
        return shared_area1
    return shared_area2

def perform_provenance(query):
    global provenance_matrix
    global visited_matrix
    global similarity_matrix
    global min_area
    global top_k 
    global same_class

    # Retrieve top-k similar
    top_similar = retrieve_topk_similar(similarity_matrix, query, k=top_k, same_class=same_class)

    # Set a threshold in the number of iteraction (queuesize)
    iteraction = max_queuesize 

    # Mark that we visited the query
    visited_matrix[query, query] = True

    while len(top_similar) > 0 and iteraction > 0:
        qs = top_similar.pop(0)

        # Ignore visited tuples
        if visited_matrix[query, qs] or  visited_matrix[qs, query]:
            continue

        # Mark tuple query, qs as visited
        visited_matrix[query, qs] = True
        visited_matrix[qs, query] = True

        try:
            if min_area:
                # Check if there is any match between qs and the query
                q_qs_shared_area = get_image_shared_area(query, qs)
                qs_q_shared_area = get_image_shared_area(qs, query)
                # Get the maximum shared after checking q->qs and qs->q
                shared_area = max_shared_area(q_qs_shared_area, qs_q_shared_area)
                # If shared_area are larger than min_area, mark query and qs in the provenance_matrix
                # We only mark this if we have never visited this pair
                if min(shared_area) > min_area:
                    provenance_matrix[query][qs] += shared_area[0]
                    provenance_matrix[qs][query] += shared_area[1]

                    # insert the top-k similar queries from qs to the top_similar list
                    top_similar += retrieve_topk_similar(similarity_matrix, qs, k=int(0.1*top_k))
            else:
                pass
        except KeyboardInterrupt:
            raise
        except Exception as e:
            pass
        iteraction -= 1

##################
#      MAIN      #   
##################

# validate args
print("Validate Arguments.")
dataset, descriptor_type, matching_strategy,\
    alignment_strategy, top_k, min_keypoints,\
    max_queuesize, min_area, same_class, dump_graph = validate_args(**vars(parse_opt()))

# Read dataset
with open(dataset) as f:
    panels_dataset = json.load(f)

##################
#  KEYPOINT DESC #   
##################
# Describe the keypoints of each image and dump them into a folder named descriptor
# If the descriptor already exists, it will not be generated again

print("Describe keypoints.")
for k, d in tqdm(panels_dataset.items()):
    describe_panel((k, d['panel_path'], descriptor_type))
print("Done.")

##################
#  Provenance    #   
##################
print("Perform Provenance Analysis.")

# Read panels embeddings
panel_image_embeddings = []
for  p_id, item in panels_dataset.items():
    panel_image_embeddings.append(item['panels_image_embedding'])
panel_image_embeddings = np.array(panel_image_embeddings)

# Create a similarity matrix from the embeddings matrix
similarity_matrix = cosine_similarity(panel_image_embeddings)

# Initialize all provenance graph as empty sets
provenance_matrix = np.zeros_like(similarity_matrix)
visited_matrix = np.zeros_like(similarity_matrix, dtype=bool)
file_name = f"results/{descriptor_type}-{matching_strategy}-{alignment_strategy}-{top_k}-{min_keypoints}-{max_queuesize}-{min_area}-{same_class}"
if os.path.isfile(file_name+".npy"):
    print(f"Provenance matrix already exists. Loading {file_name}.npy")
    provenance_matrix = np.load(file_name+".npy")
else:
    thread_map(perform_provenance,range(len(panels_dataset)), max_workers=80) 
    os.makedirs("results", exist_ok=True)
    np.save(file_name, provenance_matrix)


###############################
# Generate Graphs  Panel Level#
###############################
print("Generate Panel Level Graphs.")
predicted_graphs = panel_provenance_groups(panels_dataset, provenance_matrix, dump_graph)

# Dump json predicted graphs
os.makedirs("graphs", exist_ok=True)
os.makedirs("graphs/panel-level", exist_ok=True)
dataset_output = Path(dataset).stem
with open(f"graphs/panel-level/{dataset_output}.json", 'w') as f:
    json.dump(predicted_graphs, f, indent=4)


###############################
# Generate Graphs  Doc   Level#
###############################

doc_predicted_graphs, document_matrix = document_provenance(panels_dataset,
                                                             provenance_matrix,
                                                             dump_graph)
# Dump json predicted graphs
os.makedirs("graphs", exist_ok=True)
os.makedirs("graphs/doc-level", exist_ok=True)
dataset_output = Path(dataset).stem
with open(f"graphs/doc-level/{dataset_output}.json", 'w') as f:
    json.dump(doc_predicted_graphs, f, indent=4)
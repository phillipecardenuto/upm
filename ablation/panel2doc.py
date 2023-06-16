# import cell
import os, shutil, json
from glob import glob
import numpy as np
import networkx as nx
from pyvis.network import Network
import json



def panel2doc_provenance(predicted_panels, panel_adjacency_matrix, panels_dataset, doc_dataset, remove_graphs=True):
    """
    Convert panel level provenance to document level document
    """
    
    
    # moving panel-level to documents
    doc_level = {}
    if remove_graphs:
        for group, evs in predicted_panels.items():
            # Remove Graphs
            doc_level[group] = [e.split('/')[0] for e in evs if not "Graphs" in e]
    

        
    all_docs = sorted(list((set([ d['doc_id'] for d in doc_dataset.values()]))))

    # from panel adjenceny to doc adjencency
    panel_adjacency_matrix[panel_adjacency_matrix == -1] = 0

    doc_adjacency_matrix = np.zeros((len(doc_dataset), len(doc_dataset)))

    for i in range(len(panel_adjacency_matrix)):
        for j in range(len(panel_adjacency_matrix)):
            if panel_adjacency_matrix[i,j]:

                # If relationship is with a graph, remove it:
                if panels_dataset[str(i)]['panel_class'] == 'Graphs' or panels_dataset[str(j)]['panel_class'] == 'Graphs' :
                    continue
                if panels_dataset[str(i)]['doc_id'] == '1000-pubmed-clean':
                    doc_i = all_docs.index(panels_dataset[str(i)]['panel_path'].split("/")[-1].split("_")[0])
                else:
                    doc_i = all_docs.index(panels_dataset[str(i)]['doc_id'])

                if panels_dataset[str(j)]['doc_id'] == '1000-pubmed-clean':
                    doc_j = all_docs.index(panels_dataset[str(j)]['panel_path'].split("/")[-1].split("_")[0])
                else:
                    doc_j = all_docs.index(panels_dataset[str(j)]['doc_id'])

                doc_adjacency_matrix[doc_i,doc_j] = 1
                doc_adjacency_matrix[doc_j,doc_i] = 1


    # Connect components
    G = nx.from_numpy_array(doc_adjacency_matrix)
    # get all connected componentes of the grap
    componnets = nx.connected_components(G)

    # Insert all connected componenntes that have more than 1 node
    # into a list
    doc_provenance_graphs = []
    for cc in componnets:
        a = G.subgraph(cc)
        if a.number_of_nodes() > 1:
            doc_provenance_graphs.append(a.copy())


    # Save Predicted Document-Level graphs

    graphs = []
    for cc in range(len(doc_provenance_graphs)):
        nt = Network()
        # Load graph from nx lib
        nt.from_nx(nx.minimum_spanning_tree(doc_provenance_graphs[cc]))
        for index,node in enumerate(nt.nodes):
            # Insert doc name
            nt.nodes[index]['title'] = all_docs[(node['id'])]
        graphs.append(nt)


    # Dump Graphs in a Json file
    doc_provenance = {}

    for graph_id, graph in enumerate(graphs):
        doc_provenance['GROUP-%d'%(graph_id+1)] = {}
        doc_provenance['GROUP-%d'%(graph_id+1)] = list(set([ i['title'] for i in graph.nodes]))

        
    return doc_provenance, doc_adjacency_matrix

    
    
    
    

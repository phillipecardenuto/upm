from glob import glob
import networkx as nx
from pyvis.network import Network
import json
import numpy as np

def provenance_groups(panels_dataset, provenance_matrix, remove_chars=True):
    """
    Remove chars will remove all plot chars (graphs) from the results
    """

    # load adjacency matrix as a Graph
    G = nx.from_numpy_array(provenance_matrix)
    # get all connected componentes of the grap
    componnets = nx.connected_components(G)

    # Insert all connected componenntes that have more than 1 node
    # into a list
    provenance_graphs = []
    for cc in componnets:
        a = G.subgraph(cc)
        if a.number_of_nodes() > 1:
            provenance_graphs.append(a.copy())


    # For each connected component from provenance_analisys create a visualization of the graph
    graphs = []
    for cc in range(len(provenance_graphs)):
        nt = Network()
        # Load graph from nx lib
        nt.from_nx(nx.minimum_spanning_tree(provenance_graphs[cc]))
        for index,node in enumerate(nt.nodes):
            # Insert image as node
            imgpath = panels_dataset[str(node['id'])]['panel_path']
            nt.nodes[index]['title'] = panels_dataset[str(node['id'])]['doc_id']
            imgpath = imgpath.split('stock-photo-papermill/')[1] if 'stock-photo-papermill' in imgpath \
                         else imgpath.split('papermill-datasets/')[1]
            nt.nodes[index]['image'] = imgpath
        
        graphs.append(nt)


    # Dump Graphs in a Json file
    predicted_graphs = {}

    index = 0
    for graph_id, graph in enumerate(graphs):
        if remove_chars:
            nodes = [ i['image'] for i in graph.nodes if 'graphs' not in i['image'].lower()]
            
        if len(nodes) > 1:
            index+=1
            predicted_graphs['GROUP-%d'%(index)] = nodes

    return predicted_graphs
import os
import numpy as np
import networkx as nx
from pyvis.network import Network
import base64
from io import BytesIO
from PIL import Image

VISUALIZATION_OPTIONS="""
    var options = {
      "edges": {
        "arrows": {
          "middle": {
            "enabled": false
          }
        },
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "layout": {
        "hierarchical": {
          "enabled": true
        }
      },
      "physics": {
      "enable": false,
        "hierarchicalRepulsion": {
          "centralGravity": 0
        },
        "minVelocity": 0.75,
        "solver": "hierarchicalRepulsion"
      }
    }
    """

def get_connected_components(matrix):
    """Get connected components from a matrix
    """
    G = nx.from_numpy_array(matrix)
    # get all connected componentes of the grap
    components = nx.connected_components(G)

    # Insert all connected components that have more than 1 node
    # into a list
    graphs = []
    for cc in components:
        graph = G.subgraph(cc)
        if graph.number_of_nodes() > 1:
            graphs.append(graph.copy())

    return graphs

def get_base64_encoded_image(image_path, size=300):

    # Open Image and resize to 'size'
    img = Image.open(image_path)
    width, height = img.size
    if width > height:
        ratio = height / width
        width = size
        height = int(ratio *width )
    else:
        ratio = width / height
        height = size
        width = int(ratio *height )

    img = img.resize((width, height))

    # Encode to base64
    imagefile = BytesIO()
    img.save(imagefile, format='JPEG')

    return base64.b64encode(imagefile.getvalue()).decode('utf-8')

def panel_provenance_groups(panels_dataset, shared_area_matrix,
                            dump_graphs=False):

    # load adjacency matrix as a Graph
    provenance_graphs = get_connected_components(shared_area_matrix)

    # For each connected component from provenance 
    # analysis create a visualization of the graph
    
    predicted_graphs_json = {}

    for cc in range(len(provenance_graphs)):
        nt = Network('1024px','2000px', notebook=True, heading="GRAPH-%d"%cc, cdn_resources='in_line')
        nt.show_buttons(filter_=   ['configure','layout','interaction','physics','edges'])
        # nt.set_options(options)
        # Load graph from nx lib
        nt.from_nx(nx.maximum_spanning_tree(provenance_graphs[cc]))

        predicted_graphs_json[f'GROUP-{cc+1}'] = {}
        panel_group = []
        for index,node in enumerate(nt.nodes):
            panel_group.append(panels_dataset[str(node['id'])]['panel_path'])
        predicted_graphs_json[f'GROUP-{cc+1}'] =  sorted(list(set(panel_group)))
        

        if dump_graphs:
          for index,node in enumerate(nt.nodes):
              # Insert image as node
              imgpath = panels_dataset[str(node['id'])]['panel_path']
              nt.nodes[index]['title'] = panels_dataset[str(node['id'])]['doc_id']
              nt.nodes[index]['image'] = 'data:image/jpeg;base64,{}'.format(get_base64_encoded_image(imgpath))
              nt.nodes[index]['shape'] = 'image'
              nt.nodes[index]['label'] = ' ' #panels_dataset[str(node['id'])]['panel_path']
          # Save html file
          os.makedirs(f'graphs/',exist_ok=True)
          nt.show(f"graphs/group-panels-{cc+1}.html")

    return predicted_graphs_json


def document_provenance(panels_dataset, panel_matrix, dump_graphs=False):
    """Calculate document level provenance matrix from panel 
    level provenance matrix
    """

    # documents set
    all_docs = set()
    for docs in panels_dataset.values():
        all_docs.add(docs['doc_id'])
    all_docs = sorted(list(all_docs))
    n_docs = len(all_docs)
    document_matrix = np.zeros((n_docs, n_docs))

    # for each pair of panels that share area
    # indicate their source document in the document matrix
    for i in range(panel_matrix.shape[0]):
        for j in range(panel_matrix.shape[1]):
            if panel_matrix[i,j]:
                document_matrix[all_docs.index(panels_dataset[str(i)]['doc_id']),
                                all_docs.index(panels_dataset[str(j)]['doc_id'])] +=1
                document_matrix[all_docs.index(panels_dataset[str(j)]['doc_id']),
                                all_docs.index(panels_dataset[str(i)]['doc_id'])] +=1


    doc_provenance_graph = get_connected_components(document_matrix)
    # Document group
    graphs = []
    for cc in range(len(doc_provenance_graph)):
      nt = Network('1024px','2000px', notebook=True, heading="GRAPH-%d"%cc, cdn_resources='in_line')
      # Load graph from nx lib
      nt.from_nx(nx.maximum_spanning_tree(doc_provenance_graph[cc]))
      for index,node in enumerate(nt.nodes):
        # Insert doc name
        nt.nodes[index]['title'] = all_docs[(node['id'])]

      if dump_graphs:
        # Save html file
        os.makedirs(f'graphs/',exist_ok=True)
        nt.show(f"graphs/group-docs-{cc+1}.html")

      graphs.append(nt)

    predicted_graphs_json = {}

    for graph_id, graph in enumerate(graphs):
      predicted_graphs_json['GROUP-%d'%(graph_id+1)] = {}
      predicted_graphs_json['GROUP-%d'%(graph_id+1)] = list(set([ i['title'] for i in graph.nodes]))

    return predicted_graphs_json, document_matrix
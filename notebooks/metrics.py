from re import A
import pandas as pd
import json

def f1(p,r):
    if p==0 and r==0:
        return 0
    return 2*(p*r) / (p+r)

def matrix2panel(panel_id, panel_dataset):
    """
    Return all info from a panel within the dataset, using the panel_id
    
    Each row from the predicted matrix represent a panel id from the dataset
    """
    return panel_dataset[str(panel_id)]

def find_annotated_panel_group(panel, annotated_groups):
    """
    Finds the annotated group of a panel.
    If the panel was not in the annotation return ""
    """
    for group, item in annotated_groups.items():
        for annotated_panels in item['nodes']:
            if panel == annotated_panels['panel']:
                return group
    return ""

def get_annotated_panel_id(group, panel):
    for node in group['nodes']:
        if panel == node['panel']:
            return node['id']
        
def panel_to_dataset_panel_id(panel, panel_dataset):
    for _id, items in panel_dataset.items():
        if panel in items['panel_path']:
            return int(_id)
        
def annotated_panel_id_to_panel_name(group, panel_id):
    for node in group['nodes']:
        if panel_id == node['id']:
            return node['panel']
        
def annotated_links(annotated_group, panel, annotated_groups):
    """
    Return all related panels linked to 'panel'
    """
    
    group = annotated_groups[annotated_group]
    panel_id = get_annotated_panel_id(group, panel)
    
    # get all others linked id with panel_id
    linked_panels = []
    for link in group['links']:
        if panel_id == link[0]:
            linked_panels.append(link[1])
            
        if panel_id == link[1]:
            linked_panels.append(link[0])
            
    # Translate annotated panel id to panel path
    linked_panels = [annotated_panel_id_to_panel_name(group, p) for p in linked_panels]
    return linked_panels

def predicted_links(panel, predicted_adj_matrix, panel_dataset):
    """
    Return all related panels linked to a predicted 'panel'
    """

    # Get panel id
    panel_id = panel_to_dataset_panel_id(panel, panel_dataset)
    # Get row in the predicted matrix
    linked_panels = predicted_adj_matrix[panel_id]
    
    # Get the indexes of all nonzeros values
    linked_panels = linked_panels.nonzero()[0]
    
    linked_panels = [panel_dataset[str(p)]['panel_path'] for p in linked_panels]
    # Keep the format of the name
    # linked_panels =  [o.replace("\n", "").split('datasets/stock-photo-papermill/')[1] 
    #                   if 'stock-photo-papermill' in o
    #                   else o.replace("\n", "").split('datasets/1000-pubmed-clean/')[1]
    #                   for o in linked_panels]
    return linked_panels
    
#########################
# Edge Precision
#########################
def relationship_precision (panel, annotated_groups, panel_dataset, predicted_adj_matrix):
    """
    Given a predicted panel, find the precision of its relationship 
    """
    # check if the predicted panel is in the gt (annotated_groups), if not return 0
    annotated_panel_group = find_annotated_panel_group(panel, annotated_groups)
    if not annotated_panel_group:
        return 0
    
    # get all panels that was annotated with a relationship with the input panel 
    annotated_related_panels = annotated_links(annotated_panel_group, panel, annotated_groups)
    
    # get all predicted linked panels
    predicted_related_panels = predicted_links(panel, predicted_adj_matrix, panel_dataset)

    # Relationship precision
    rp = len(set(annotated_related_panels).intersection(set(predicted_related_panels))) / len(predicted_related_panels)
    
    return rp
    
###########################################################
###########################################################
#########################
# Edge  RECALL
#########################
def find_predicted_panel_group(panel, predicted_groups):
    """
    Finds the annotated group of a panel.
    If the panel was not in the annotation return ""
    """
    for group, items in predicted_groups.items():
        for predicted_panels in items:
            if panel == predicted_panels:
                return group
    return ""

def relationship_recall(panel, predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix):
    """
    Given an annotated panel, find the recall of its relationship 
    """
    
    # check if panel was predicted with a relationship
    predicted_group = find_predicted_panel_group(panel, predicted_groups)
    if not predicted_group:
        return 0
    
    # get all predicted linked panels
    predicted_related_panels = predicted_links(panel, predicted_adj_matrix, panel_dataset)
    
    # get annotated group
    annotated_panel_group = find_annotated_panel_group(panel, annotated_groups)
    
    # get all panels that was annotated with a relationship with the input panel 
    annotated_related_panels = annotated_links(annotated_panel_group, panel, annotated_groups)
    
    # Relationship recall
    rr = len(set(annotated_related_panels).intersection(set(predicted_related_panels))) / len(annotated_related_panels)
    
    return rr
###########################################################
###########################################################

#########################
# Edge F1-Measure
#########################

def relationship_quality(panel, predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix):
    """
    Given a predicted panel, find its f1-measure
    """
    precision = relationship_precision (panel, annotated_groups, panel_dataset, predicted_adj_matrix)
    recall = relationship_recall(panel, predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix)
    
    return f1(precision, recall)

###########################################################
###########################################################

################
# Node Precision
################

def group_precision (panel, predicted_groups, annotated_groups ):
    """
    Given a predicted panel, find the precision of nodes within the same group 
    """
    
    # check if the predicted panel is in the gt, if not return 0
    annotated_panel_group = find_annotated_panel_group(panel, annotated_groups)
    if not annotated_panel_group:
        return 0
    
    # get all panels that was annotated within the same group as 'panel'
    annotated_group = annotated_groups[annotated_panel_group]
    annotated_group = [ i['panel'] for i in annotated_group['nodes']]
    
    # get all predicted panels within the same group as panel
    predicted_panel_group = find_predicted_panel_group(panel, predicted_groups)
    predicted_group = predicted_groups[predicted_panel_group]
    
    # Relationship precision
    gp = len(set(predicted_group).intersection(set(annotated_group))) / len(predicted_group)
    
    return gp

# Node Recall
def group_recall (panel, predicted_groups, annotated_groups):
    """
    Given a annotated panel, find the recall of nodes within the same group 
    """
    
    # check if the annotated panel was predicted, if not return 0
    predicted_panel_group = find_predicted_panel_group(panel, predicted_groups)
    if not predicted_panel_group:
        return 0
    
    # get all predicted panels within the same group as panel
    predicted_group = predicted_groups[predicted_panel_group]

    # get all panels that was annotated within the same group as 'panel'
    annotated_panel_group = find_annotated_panel_group(panel, annotated_groups)
    annotated_group = annotated_groups[annotated_panel_group]
    annotated_group = [ i['panel'] for i in annotated_group['nodes']]
    
    gr = len(set(predicted_group).intersection(set(annotated_group))) / len(annotated_group)
    
    return gr


############################################################
############################################################

###############
# Joined Groups
###############
"""
Considering that some groups of panels might be predicted split,
we need to join the groups to have a better notion of the results.
"""

def joined_group_precision (panel, predicted_groups, annotated_groups):
    """
    Given a predicted panel, find the precision of nodes within the same group 
    """
    
    # check if the predicted panel is in the gt, if not return 0
    annotated_panel_group = find_annotated_panel_group(panel, annotated_groups)
    if not annotated_panel_group:
        return 0
    
    # get all predicted panels within the same group as panel
    predicted_panel_group = find_predicted_panel_group(panel, predicted_groups)
    predicted_group = predicted_groups[predicted_panel_group]
    
    # Join all annotated groups that shares a panel with the
    # predicted group
    joined_annotated_groups = set()
    for p in predicted_group:
        annotated_panel_group = find_annotated_panel_group(p, annotated_groups)
        if annotated_panel_group:
            # get all panels that was annotated within the same group as 'panel'
            annotated_group = annotated_groups[annotated_panel_group]
            annotated_group = [ i['panel'] for i in annotated_group['nodes']]
            for item in annotated_group:
                joined_annotated_groups.add(item)
            
    
    # Joined Group Precision
    gp = len(set(predicted_group).intersection(set(joined_annotated_groups))) / len(predicted_group)
    
    return gp

def joined_group_recall (panel, predicted_groups, annotated_groups):
    """
    Given an annotated panel, find the recall of nodes within the same group 
    """
    
    # check if the annotated panel was predicted, if not return 0
    predicted_panel_group = find_predicted_panel_group(panel, predicted_groups)
    if not predicted_panel_group:
        return 0
    
    # get all panels that was annotated within the same group as 'panel'
    annotated_panel_group = find_annotated_panel_group(panel, annotated_groups)
    annotated_group = annotated_groups[annotated_panel_group]
    annotated_group = [ i['panel'] for i in annotated_group['nodes']]
    
    
     # Join all annotated groups that shares a panel with the
    # predicted group
    joined_predicted_groups = set()
    for p in annotated_group:
        predicted_panel_group = find_predicted_panel_group(p, predicted_groups)
        if predicted_panel_group:
            # get all panels that was predicted within the same group as 'panel'
            predicted_panels = predicted_groups[predicted_panel_group]
            for item in predicted_panels:
                joined_predicted_groups.add(item)

    # Joined Group Recall
    gr = len(set(annotated_group).intersection(set(joined_predicted_groups))) / len(annotated_group)
    
    return gr


###########################################################
###########################################################

### GLOBAL METRICS

def global_precision_per_class(predicted_groups, annotated_groups, _class):
    """
    Finds the global precision of the predicted panels
    """

    true_positives = 0
    false_positives = 0

    for panels in predicted_groups.values():
        for panel in panels:
            if not _class.lower() in panel.lower():
                continue
            if find_annotated_panel_group(panel, annotated_groups):
                true_positives+=1
            else:
                false_positives +=1
    if true_positives == 0:
        return 0
    precision = true_positives / (true_positives + false_positives)
    return precision

def global_recall_per_class(predicted_groups, annotated_groups, _class):
    """
    Finds the global recall of the predicted panels
    """

    true_positives = 0
    false_negatives = 0

    for panels in annotated_groups.values():
        for panel in panels['nodes']:
            panel = panel['panel']
            if not _class.lower() in panel.lower():
                continue
            if find_predicted_panel_group(panel, predicted_groups):
                true_positives+=1
            else:
                false_negatives+=1
    if true_positives == 0:
        return 0
    recall = true_positives / (true_positives + false_negatives)
    return recall 

def global_precision(predicted_groups, annotated_groups, ignore_graphs=False):
    """
    Finds the global precision of the predicted panels
    """

    true_positives = 0
    false_positives = 0

    for panels in predicted_groups.values():
        for panel in panels:
            if ignore_graphs:
                if 'graphs' in panel.lower():
                    continue
            if find_annotated_panel_group(panel, annotated_groups):
                true_positives+=1
            else:
                false_positives +=1
    if true_positives == 0:
        return 0

    precision = true_positives / (true_positives + false_positives)
    return precision

def global_recall(predicted_groups, annotated_groups, ignore_graphs=False):
    """
    Finds the global recall of the predicted panels
    """

    true_positives = 0
    false_negatives = 0

    for panels in annotated_groups.values():
        for panel in panels['nodes']:
            panel = panel['panel']
            if ignore_graphs:
                if 'graphs' in panel.lower():
                    continue
            if find_predicted_panel_group(panel, predicted_groups):
                true_positives+=1
            else:
                false_negatives+=1

    if true_positives == 0:
        return 0
    recall = true_positives / (true_positives + false_negatives)
    return recall 



#############################################################
############################################################

######################
#  Perform Evaluation#
#  For all Dataset   #
######################
def rp_eval(predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix):
    """
    Perform the precision relationship evaluation for all panels within the predicted set

    RETURN:
    ------
    Return a pandas dataframe with all predicted panels (as suspected) with their respective relationship precision
    """
    precision_df = pd.DataFrame()
    index = 0
    for group, nodes in predicted_groups.items():
        for panel in nodes:
            index += 1
            # precision_df.loc[index,'Class'] = panel.split('-')[-1][:-4].split("_")[-1]
            precision_df.loc[index,'Class'] = panel.split('/')[2]
            precision_df.loc[index,'Panel'] = panel
            precision_df.loc[index,'Precision'] = relationship_precision (panel, annotated_groups, panel_dataset, predicted_adj_matrix)
    
    return precision_df
        
def rr_eval(predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix):
    """
    Perform the recall relationship evaluation for all panels considering the predicted set
    RETURN:
    ------
    Return a pandas dataframe with all annotated panels  with their respective relationship recall
    """
    recall_df = pd.DataFrame()
    index = 0
    for group, ann in annotated_groups.items():
        for panel in ann['nodes']:
            panel = panel['panel']
            index += 1
            recall_df.loc[index,'Recall'] = relationship_recall(panel, predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix)
            # recall_df.loc[index,'Class'] = panel.split('-')[-1][:-4].split("_")[-1]
            recall_df.loc[index,'Class'] = panel.split('/')[2]
            recall_df.loc[index,'Panel'] = panel
        
    return recall_df 

def gp_eval(predicted_groups, annotated_groups):
    """
    Perform the grouping precision evaluation for all panels within the predicted set

    RETURN:
    ------
    Return a pandas dataframe with all predicted panels (as suspected) with their respective relationship precision
    """
    precision_df = pd.DataFrame()
    index = 0
    for group, nodes in predicted_groups.items():
        for panel in nodes:
            index += 1
            # precision_df.loc[index,'Class'] = panel.split('-')[-1][:-4].split("_")[-1]
            precision_df.loc[index,'Class'] = panel.split('/')[2]
            precision_df.loc[index,'Panel'] = panel
            precision_df.loc[index,'Precision'] =joined_group_precision(panel, predicted_groups, annotated_groups)
    
    return precision_df
        
def gr_eval(predicted_groups, annotated_groups):
    """
    Perform the grouping recall evaluation for all panels considering the predicted set
    RETURN:
    ------
    Return a pandas dataframe with all annotated panels  with their respective grouping recall
    """
    recall_df = pd.DataFrame()
    index = 0
    for group, ann in annotated_groups.items():
        for panel in ann['nodes']:
            panel = panel['panel']
            index += 1
            recall_df.loc[index,'Recall'] = joined_group_recall(panel, predicted_groups, annotated_groups)
            # recall_df.loc[index,'Class'] = panel.split('-')[-1][:-4].split("_")[-1]
            recall_df.loc[index,'Class'] =  panel.split('/')[2]
            recall_df.loc[index,'Panel'] = panel
        
    return recall_df 



def panel_evaluation(predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix, _class):
    """
    Perform panel evaluation and dump the eval into a JSON file
    """

    results = {}
    # Relationship eval
    ###################
    rp = rp_eval(predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix)
    rr = rr_eval(predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix)
    
    if len(rp) ==0:
        results['relationship_precision'] = 0
        results['relationship_recall'] = 0
        results['relationship_quality'] = 0
        
    else:
        results['relationship_precision'] = rp[rp['Class'] == _class]['Precision'].mean() # Remove Graphs from eval
        results['relationship_recall'] = rr[rr['Class'] == _class]['Recall'].mean() # Remove Graphs from eval
        results['relationship_quality'] = f1(results['relationship_precision'], results['relationship_recall'])

    # Grouping eval
    ###################
    gp = gp_eval(predicted_groups, annotated_groups)
    gr = gr_eval(predicted_groups, annotated_groups)
    
    if len(gp) ==0:
        results['grouping_precision'] = 0
        results['grouping_recall'] = 0
        results['grouping_quality'] = 0
        
    else:
        results['grouping_precision'] = gp[gp['Class'] == _class]['Precision'].mean() # Remove Graphs from eval
        results['grouping_recall'] = gr[gr['Class'] == _class]['Recall'].mean() # Remove Graphs from eval
        results['grouping_quality'] = f1(results['grouping_precision'], results['grouping_recall'])

    return results

#######################################################
#  -------------------------------------------------- #
#######################################################

# Document Level Evaluation


def matrix2doc(doc_id, doc_dataset):
    """
    Return all info from a doc within the dataset, using the doc_id
    
    Each row from the predicted matrix represent a doc id from the dataset
    """
    return doc_dataset[str(doc_id)]

def find_annotated_doc_group(doc, annotated_groups):
    """
    Finds the annotated group of a document.
    If the document do not belong to any suspect group, return ""
    """
    for group, item in annotated_groups.items():
        for annotated_docs in item['nodes']:
            if doc == annotated_docs['doi']:
                return group
    return ""

def get_annotated_doc_id(group, doc):
    for node in group['nodes']:
        if doc == node['doi']:
            return node['id']
        
def doc_to_dataset_doc_id(doc, doc_dataset):
    for _id, items in doc_dataset.items():
        if doc in items['doc_id']:
            return int(_id)
        
def annotated_doc_id_to_doi(group, doc_id):
    for node in group['nodes']:
        if doc_id == node['id']:
            return node['doi']
        
def annotated_doc_links(annotated_group, doc, annotated_groups):
    """
    Return all related panels linked to 'doc'
    """
    
    group = annotated_groups[annotated_group]
    doc_id = get_annotated_doc_id(group, doc)
    
    # get all others linked id with panel_id
    linked_panels = []
    for link in group['links']:
        if doc_id == link[0]:
            linked_panels.append(link[1])
            
        if doc_id == link[1]:
            linked_panels.append(link[0])
            
    # Translate annotated doc_id to doi
    linked_panels = [annotated_doc_id_to_doi(group, p) for p in linked_panels]
    return linked_panels

def predicted_doc_links(doc, predicted_adj_matrix, doc_dataset):
    """
    Return all related docs linked to a predicted 'doc'
    """
    
    # Get doi
    doc_id = doc_to_dataset_doc_id(doc, doc_dataset)
    # Get row in the predicted matrix
    linked_panels = predicted_adj_matrix[doc_id]
    
    # Get the indexes of all nonzeros values
    linked_panels = linked_panels.nonzero()[0]
    
    
    linked_panels = [doc_dataset[str(p)]['doc_id'] for p in linked_panels]
    
    return linked_panels
    
################################
# Document Relationship Precision 
################################
    
def doc_relationship_precision(doc, annotated_groups, doc_dataset, predicted_adj_matrix):
    """
    Given a predicted doc, find the precision of its relationship 
    """
    
    # check if the predicted doc is in the gt, if not return 0
    annotated_doc_group = find_annotated_doc_group(doc, annotated_groups)
    if not annotated_doc_group:
        return 0
    
    # get all doc that was annotated with a relationship with the input panel 
    annotated_related_docs = annotated_doc_links(annotated_doc_group, doc, annotated_groups)
    
    # get all predicted linked doc
    predicted_related_docs = predicted_doc_links(doc, predicted_adj_matrix, doc_dataset)
    
    # Relationship precision
    rp = len(set(annotated_related_docs).intersection(set(predicted_related_docs))) / len(predicted_related_docs)
    
    return rp


def find_predicted_doc_group(doc, predicted_groups):
    """
    Finds the annotated group of a doc
    If the doc was not in the annotation return ""
    """
    for group, items in predicted_groups.items():
        if doc in items:
            return group
    return ""

################################
# Document Relationship Recall 
################################
def doc_relationship_recall(doc, predicted_groups, annotated_groups, doc_dataset, predicted_adj_matrix):
    """
    Given an annotated doc, find the recall of its relationship 
    """
    
    # check if panel was predicted with a relationship
    predicted_group = find_predicted_doc_group(doc, predicted_groups)
    if not predicted_group:
        return 0
    
    # get all predicted linked docs
    predicted_related_docs = predicted_doc_links(doc, predicted_adj_matrix, doc_dataset)
    
    # get annotated group
    annotated_doc_group = find_annotated_doc_group(doc, annotated_groups)
    
    # get all docs that were annotated with a relationship with the input doc 
    annotated_related_docs = annotated_doc_links(annotated_doc_group, doc, annotated_groups)
    
    # Relationship recall
    rr = len(set(annotated_related_docs).intersection(set(predicted_related_docs))) / len(annotated_related_docs)
    
    return rr

################################
# Document Relationship Quality
################################
def doc_relationship_quality(panel, predicted_groups, annotated_groups, doc_dataset, predicted_adj_matrix):
    """
    Given a predicted doc, find its f1-measure
    """
    precision = doc_relationship_precision(panel, annotated_groups, doc_dataset, predicted_adj_matrix)
    recall = doc_relationship_recall(panel, predicted_groups, annotated_groups, doc_dataset, predicted_adj_matrix)
    
    return f1(precision, recall)


################################
# Document Group Precision
################################

def doc_group_precision (doc, predicted_groups, annotated_groups ):
    """
    Given a predicted document, find the precision of nodes within the same group 
    """
    
    # check if the predicted panel is in the gt, if not return 0
    annotated_doc_group = find_annotated_doc_group(doc, annotated_groups)
    if not annotated_doc_group:
        return 0
    
    # get all docs that was annotated within the same group as 'doc'
    annotated_group = annotated_groups[annotated_doc_group]
    annotated_group = [ i['panel'] for i in annotated_group['nodes']]
    
    # get all predicted docs within the same group as panel
    predicted_doc_group = find_predicted_doc_group(doc, predicted_groups)
    predicted_group = predicted_groups[predicted_doc_group]
    
    # Relationship precision
    gp = len(set(predicted_group).intersection(set(annotated_group))) / len(predicted_group)
    
    return gp

# Node Recall
def doc_group_recall (doc, predicted_groups, annotated_groups):
    """
    Given a annotated document, find the recall of nodes within the same group 
    """
    
    # check if the annotated doc was predicted, if not return 0
    predicted_doc_group = find_predicted_doc_group(doc, predicted_groups)
    if not predicted_doc_group:
        return 0
    
    # get all predicted documents within the same group as panel
    predicted_group = predicted_groups[predicted_doc_group]

    # get all docs that was annotated within the same group as 'doc'
    annotated_doc_group = find_annotated_doc_group(doc, annotated_groups)
    annotated_group = annotated_groups[annotated_doc_group]
    annotated_group = [ i['panel'] for i in annotated_group['nodes']]
    
    gr = len(set(predicted_group).intersection(set(annotated_group))) / len(annotated_group)
    
    return gr

############################################################
############################################################

###############
# Joined Groups
###############
"""
Considering that some groups of doc might be predicted split,
we need to join the groups to have a better notion of the results.
"""
def doc_joined_group_precision (doc, predicted_groups, annotated_groups):
    """
    Given a predicted doc, find the precision of nodes within the same group 
    """
    
    # check if the predicted panel is in the gt, if not return 0
    annotated_doc_group = find_annotated_doc_group(doc, annotated_groups)
    if not annotated_doc_group:
        return 0
    
    # get all predicted docs within the same group as panel
    predicted_doc_group = find_predicted_doc_group(doc, predicted_groups)
    predicted_group = predicted_groups[predicted_doc_group]
    
    # Join all annotated groups that shares a panel with the
    # predicted group
    joined_annotated_groups = set()
    for p in predicted_group:
        annotated_doc_group = find_annotated_doc_group(p, annotated_groups)
        if annotated_doc_group:
            # get all panels that was annotated within the same group as 'panel'
            annotated_group = annotated_groups[annotated_doc_group]
            annotated_group = [ i['doi'] for i in annotated_group['nodes']]
            for item in annotated_group:
                joined_annotated_groups.add(item)
            
    # Joined Group Precision
    gp = len(set(predicted_group).intersection(set(joined_annotated_groups))) / len(predicted_group)
    
    return gp

def doc_joined_group_recall (doc, predicted_groups, annotated_groups):
    """
    Given an annotated doc, find the recall of nodes within the same group 
    """
    
    # check if the annotated doc was predicted, if not return 0
    predicted_doc_group = find_predicted_doc_group(doc, predicted_groups)
    if not predicted_doc_group:
        return 0
    
    # get all docs that was annotated within the same group as 'doc'
    annotated_doc_group = find_annotated_doc_group(doc, annotated_groups)
    annotated_group = annotated_groups[annotated_doc_group]
    annotated_group = [ i['doi'] for i in annotated_group['nodes']]
    
    
     # Join all annotated groups that shares a doc with the
    # predicted group
    joined_predicted_groups = set()
    for p in annotated_group:
        predicted_doc_group = find_predicted_doc_group(p, predicted_groups)
        if predicted_doc_group:
            # get all panels that was predicted within the same group as 'panel'
            predicted_docs = predicted_groups[predicted_doc_group]
            for item in predicted_docs:
                joined_predicted_groups.add(item)

    # Joined Group Recall
    gr = len(set(annotated_group).intersection(set(joined_predicted_groups))) / len(annotated_group)
    
    return gr



###########################################################
### GLOBAL METRICS
###########################################################

def doc_global_precision(predicted_groups, annotated_groups):
    """
    Finds the global precision of the predicted panels
    """

    true_positives = 0
    false_positives = 0

    for docs in predicted_groups.values():
        for doc in docs:
            if find_annotated_doc_group(doc, annotated_groups):
                 true_positives+=1
            else:
                false_positives +=1
    
    if true_positives + false_positives == 0:
        return 0
    precision = true_positives / (true_positives + false_positives)
    return precision

def doc_global_recall(predicted_groups, annotated_groups):
    """
    Finds the global recall of the predicted panels
    """

    true_positives = 0
    false_negatives = 0

    for docs in annotated_groups.values():
        for doc in docs['nodes']:
            doc = doc['doi']
            if find_predicted_doc_group(doc, predicted_groups):
                true_positives+=1
            else:
                false_negatives+=1
    if true_positives + false_negatives == 0:
        return 0
    recall = true_positives / (true_positives + false_negatives)
    return recall 


###########################################################
###########################################################

######################
#  Perform Evaluation#
#  For all Dataset   #
######################
def doc_rp_eval(predicted_groups, annotated_groups, doc_dataset, predicted_adj_matrix):
    """
    Perform the precision relationship evaluation for all docs within the predicted set

    RETURN:
    ------
    Return a pandas dataframe with all predicted docs (as suspected) with their respective relationship precision
    """
    precision_df = pd.DataFrame()
    index = 0
    for group, nodes in predicted_groups.items():
        for doc in nodes:
            index += 1
            precision_df.loc[index,'Document'] = doc 
            precision_df.loc[index,'Precision'] = doc_relationship_precision (doc, annotated_groups, doc_dataset, predicted_adj_matrix)
    
    return precision_df
        
def doc_rr_eval(predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix):
    """
    Perform the recall relationship evaluation for all panels considering the predicted set
    RETURN:
    ------
    Return a pandas dataframe with all annotated panels  with their respective relationship recall
    """
    recall_df = pd.DataFrame()
    index = 0
    for group, ann in annotated_groups.items():
        for doc in ann['nodes']:
            doc = doc['doi']
            index += 1
            recall_df.loc[index,'Recall'] = doc_relationship_recall(doc, predicted_groups, annotated_groups, panel_dataset, predicted_adj_matrix)
            recall_df.loc[index,'Panel'] = doc 
        
    return recall_df 

def doc_gp_eval(predicted_groups, annotated_groups):
    """
    Perform the grouping precision evaluation for all doc within the predicted set

    RETURN:
    ------
    Return a pandas dataframe with all predicted doc (as suspected) with their respective relationship precision
    """
    precision_df = pd.DataFrame()
    index = 0
    for group, nodes in predicted_groups.items():
        for doc in nodes:
            index += 1
            precision_df.loc[index,'Document'] = doc 
            precision_df.loc[index,'Precision'] =doc_joined_group_precision(doc, predicted_groups, annotated_groups)
    
    return precision_df
        
def doc_gr_eval(predicted_groups, annotated_groups):
    """
    Perform the grouping recall evaluation for all doc considering the predicted set
    RETURN:
    ------
    Return a pandas dataframe with all annotated doc with their respective grouping recall
    """
    recall_df = pd.DataFrame()
    index = 0
    for group, ann in annotated_groups.items():
        for doc in ann['nodes']:
            doc = doc['doi']
            index += 1
            recall_df.loc[index,'Recall'] = doc_joined_group_recall(doc, predicted_groups, annotated_groups)
            recall_df.loc[index,'Document'] = doc 
        
    return recall_df 


def doc_evaluation(predicted_groups, annotated_groups, doc_dataset, predicted_adj_matrix):
    """
    Perform panel evaluation and dump the eval into a JSON file
    """

    results = {}
    # Relationship eval
    ###################
    rp = doc_rp_eval(predicted_groups, annotated_groups, doc_dataset, predicted_adj_matrix)
    rr = doc_rr_eval(predicted_groups, annotated_groups, doc_dataset, predicted_adj_matrix)
    
    if len(rp):
        results['relationship_precision'] = rp["Precision"].mean() # Remove Graphs from eval
        results['relationship_recall'] = rr["Recall"].mean()
        results['relationship_quality'] = f1(results['relationship_precision'], results['relationship_recall'])
    else:
        results['relationship_precision'] =   results['relationship_recall'] = results['relationship_quality'] = 0

    # Grouping eval
    ###################
    gp = doc_gp_eval(predicted_groups, annotated_groups)
    gr = doc_gr_eval(predicted_groups, annotated_groups)
    
    if len(gp):
        results['grouping_precision'] = gp['Precision'].mean() # Remove Graphs from eval
        results['grouping_recall'] = gr['Recall'].mean() # Remove Graphs from eval
        results['grouping_quality'] = f1(results['grouping_precision'], results['grouping_recall'])
    else:
        results['grouping_precision'] = results['grouping_recall'] = results['grouping_quality'] = 0

    # Global Eval
    #############
    results['global_precision'] = doc_global_precision(predicted_groups, annotated_groups)
    results['global_recall'] = doc_global_recall(predicted_groups, annotated_groups)
    results['global_quality'] = f1(results['global_precision'], results['global_recall'])

    return results
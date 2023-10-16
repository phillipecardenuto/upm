"""
This file reproduce the method of Acuna et al Bioscience-scale automated detection of figure element reuse
Some parameters were not mentioned in the article, so we could not reproduce the exact same algorithm 
presented by Acuna et al work.

Atuhors: Daniel Moreira and Phillipe Cardenuto - December 2021
"""

import numpy as np
EPISLON = np.finfo(float).eps

import cv2
import math
from time import time

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph




def _clusterize_keypoints(keypoints, descriptions, image_shape,
                          conn_neighbor_rate=0.1, dist_thresh_rate=0.001, cpu_count=-1):
    positions = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

    nb_count = int(round(len(positions) * conn_neighbor_rate))

    """
    @ Joao Comment
    Checking if there is any neighboor to cluster, if there is not return
    """
    if nb_count == 0:
        if len(keypoints) > 0:
            return [(keypoints,descriptions)]
        else:
            return[([],[])]
    dist_thresh = image_shape[0] * image_shape[1] * dist_thresh_rate

    forced_conn = kneighbors_graph(X=positions, n_neighbors=nb_count, n_jobs=cpu_count)
    clustering = AgglomerativeClustering(n_clusters=None, connectivity=forced_conn, distance_threshold=dist_thresh)
    clustering.fit(positions)

    labels = clustering.labels_

    clusters = {}
    for i in range(0, len(labels)):
        label = labels[i]
        if label not in clusters.keys():
            clusters[label] = ([], [])

        clusters[label][0].append(keypoints[i])
        clusters[label][1].append(descriptions[i])

    return [clusters[label] for label in clusters.keys()]

def sift_detect_sift_describe(image, kp_count=1000, eps=1e-7):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SIFT detector and descriptor
    sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures=kp_count)

    # detects SIFT keypoints over the given image
    keypoints = sift_detector.detect(image)
    keypoints = list(keypoints)

    # if no keypoints were detected, returns empty keypoints and descriptions
    if len(keypoints) == 0:
        return [], []

    # else...
    # sorts the obtained keypoints according to their response, and keeps only the top-<kp_count> ones
    
    keypoints.sort(key=lambda k: k.response, reverse=True)
    del keypoints[kp_count:]

    # describes the remaining keypoints
    keypoints, descriptions = sift_detector.compute(image, keypoints)

    # returns the obtained keypoints and their respective descriptions
    return keypoints, descriptions


# Performs G2NN selection of the given two sets of keypoints and their respective descriptions
# (<keypoints1>, <descriptions1>), (<keypoints2>, <descriptions2>). Please provide <within_same_image>
# as TRUE in the case of the two keypoint sets coming from the same (single) image.
# Parameter <k_rate> in [0.0, 1.0] helps to define how many neighbors are matched to each given keypoint.
# Parameter <nndr_threshold> in [0.0, 1.0] is the maximum value used to consider a match useful, according to
# its difference (distance-wise) to the next closest match (G2NN principle).
# Returns the selected keypoints and respective descriptions for each one of the given two sets of keypoints.
def _g2nn_keypoint_selection(keypoints1, descriptions1, keypoints2, descriptions2, k_rate=0.5, nndr_threshold=0.60):
    # defines the two sets of keypoints to be matched
    # (smaller set: keypoints1, larger set: keypoints2)
    swapped = False
    if len(keypoints2) < len(keypoints1):
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1
        swapped = True

    # matches keypoints1 towards keypoints2
    knn_matches = cv2.BFMatcher().knnMatch(descriptions1, descriptions2, k=int(round(len(keypoints1) * k_rate)))

    # g2NN match selection
    selected_matches = []
    for _, matches in enumerate(knn_matches):
        for i in range(0, len(matches) - 1):
            """@Joao Comment
            Next, we include the matches only if the distance of the nearest neighbor is 60%
            or less than the distance of the second nearest neighbor
            
            Also, we add an Epislon to avoid 0 division
            """
            ratio = matches[i].distance / (matches[i + 1].distance +EPISLON)
            
              
            """
            There is a statement of Acuna et al that removes matches with 40pix or less apart
            Not sure how to implement this, since the matche would be across figures.

            pt1 = np.array(keypoints1[matches[i].queryIdx].pt)
            pt2 = np.array(keypoints2[matches[i].trainIdx].pt)
            np.linalg.norm((pt1, pt2)) >= 40
            """
            if ratio <= nndr_threshold:
                """@ Joao Comment
                Paper Amerini et al. suggest the thresold to be 0.5
                """
                selected_matches.append(matches[i])
            else:
                break

    # keypoint and description selection, based on selected matches
    indices1 = []
    indices2 = []
    for match in selected_matches:
        if match.queryIdx not in indices1:
            indices1.append(match.queryIdx)

        if match.trainIdx not in indices2:
            indices2.append(match.trainIdx)
            
    selected_keypoints_1 = []
    selected_descriptions_1 = []
    for index in indices1:
        selected_keypoints_1.append(keypoints1[index])
        selected_descriptions_1.append(descriptions1[index])

    selected_keypoints_2 = []
    selected_descriptions_2 = []
    for index in indices2:
        selected_keypoints_2.append(keypoints2[index])
        selected_descriptions_2.append(descriptions2[index])

    # undoes swapping, if it is the case
    if swapped:
        selected_keypoints_1, selected_keypoints_2 = selected_keypoints_2, selected_keypoints_1
        selected_descriptions_1, selected_descriptions_2 = selected_descriptions_2, selected_descriptions_1

    # returns the selected keypoints
    return selected_keypoints_1, selected_descriptions_1, selected_keypoints_2, selected_descriptions_2

def is_small_cluster(cluster, area_th=40*40):
    """
    Following the guideline from Acuna et al,
    this function return a boolean value saying if the input cluster
    has area smaller than area_th
    """
    points = np.array([ p.pt for p in cluster[0]])
    x0 = np.min(points[:,0])
    x1 = np.max(points[:,0])
    y0 = np.min(points[:,1])
    y1 = np.max(points[:,1])
    area = (x1 - x0) * (y1 - y0)
    return area < area_th
        

# Clusterizes the given <keypoints> and respective <descriptions> according to their (x, y) positions
# within the source image. The shape (height, width) <image_shape> of the source image must also be given.
# Additional clustering configuration parameters:
# <conn_neighbor_rate> - Rate in [0.0, 1.0] to define the number of keypoints used to lock connectivity in
# the clustering solution;
#   <dist_thresh_rate> - Rate in [0.0, 1.0] to define the distance threshold used in the clustering solution;
#          <cpu_count> - Number of CPU cores used in clustering. Give -1 to use all cores.
# Returns the computed clusters as a list of pairs of lists (one pair for each cluster),
# whose 1st element of each pair is a list of keypoints, and whose 2nd element is the respective list of descriptions.
def _clusterize_keypoints(keypoints, descriptions, image_shape,
                          conn_neighbor_rate=0.1, dist_thresh=30, cpu_count=-1):
    positions = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

    nb_count = int(round(len(positions) * conn_neighbor_rate))

    """
    @ Joao Comment
    Checking if there is any neighboor to cluster, if there is not return
    """
    if nb_count == 0:
        if len(keypoints) > 0:
            return [(keypoints,descriptions)]
        else:
            return[([],[])]

    forced_conn = kneighbors_graph(X=positions, n_neighbors=nb_count, n_jobs=cpu_count)
    clustering = AgglomerativeClustering(n_clusters=None,
                                         connectivity=forced_conn,
                                         distance_threshold=dist_thresh,
                                         linkage='single')
    clustering.fit(positions)

    labels = clustering.labels_

    clusters = {}
    for i in range(0, len(labels)):
        label = labels[i]
        if label not in clusters.keys():
            clusters[label] = ([], [])

        clusters[label][0].append(keypoints[i])
        clusters[label][1].append(descriptions[i])
    
    """
    Discard Clusters with area less than 40x40 (from Acuna)
    """
    for i in range(len(labels)):
        if clusters.get(label):
            if is_small_cluster(clusters[label]):
                del clusters[label]
    
    return [clusters[label] for label in clusters.keys()]



# Computes, from the given two sets of keypoints and their respective descriptions (<keypoints1>, <descriptions1>),
# (<keypoints2>, <descriptions2>), the largest set of matches that are geometrically consistent among themselves.
# Parameter <nndr_thresh> in [0.0, 1.0] helps to select useful keypoints (2NN, a.k.a. Lowe's NNDR selection).
# Returns a list of matches of the OpenCV DMatch type.
def _consistent_match(keypoints1, descriptions1, keypoints2, descriptions2, nndr_thresh=0.85):
    # defines the two sets of keypoints to be matched
    # (smaller set: keypoints1, larger set: keypoints2)
    swapped = False
    if len(keypoints2) < len(keypoints1):
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1
        swapped = True

    # matches keypoints1 towards keypoints2 and performs 2NN (NNDR) verification
    knn_matches = cv2.BFMatcher().knnMatch(descriptions1, descriptions2, k=2)
    good_matches = []

    """
    @ Joao comment
    if the number of knn matches is less than 3, 
    don't check its distance.
    """
    if len(knn_matches) > 3:
        for i, (a, b) in enumerate(knn_matches):
            if b.distance != 0.0 and (a.distance / b.distance) < nndr_thresh:
                good_matches.append(a)
    else:
        good_matches = [a for a, b in knn_matches ]
    
    if len(good_matches) == 0:
        return []
    good_matches.sort(key=lambda m: m.distance)
    

    # Apply affine transformation in the good_matches
    src_points = np.array([keypoints1[p.queryIdx].pt for p in  good_matches ])
    dest_points = np.array([keypoints2[p.trainIdx].pt for p in  good_matches ])
        
    """
    Also, if the sheer of the estimated transformation is more than 15 degrees, remove the matched cluster
    """
    # Check angle Transformation
    try:
        (H, mask) = cv2.findHomography(src_points, dest_points, cv2.RANSAC)
    
        u, _, vh = np.linalg.svd(H[0:2, 0:2])
        R = u @ vh
        angle = math.atan2(R[1,0], R[0,0])  *180 / np.pi
        angle = angle if angle <= 90 else abs(180 - angle)
        
    except:
        return []
    
    
    """
    If less than 80% of the keypoints are used by the RanSac algorithm, we will remove the matched cluster.
    Also, if the sheer of the estimated transformation is more than 15 degrees, remove the matched cluster.
    """
    if (sum(mask) / len(src_points)) < 0.8:
        return []
    if angle > 15:
        return []
        
    # prepares the selected matches to be returned
    answer = []
    for i in range(len(good_matches)):
        # fixes swap, if needed
        if swapped:
            good_matches[i].queryIdx, good_matches[i].trainIdx = good_matches[i].trainIdx, good_matches[i].queryIdx

        answer.append(good_matches[i])

    answer.sort(key=lambda m: m.distance)
    return answer


# Conciliates the given set of match clusters <clustered_matches>, based on their image-space transformation agreement.
# Parameters <angle_diff_tolerance> (in radians) and <dist_diff_tolerance> (in pixels) define the tolerance to
# consider two custer equivalents in terms of transformations.
# Returns the conciliated (merged) list of clusters.
def _conciliate_clusters(clustered_matches, angle_diff_tolerance=0.2356194, dist_diff_tolerance=40):
    # computes the average transformation for each cluster
    transforms = []
    for cluster in clustered_matches:
        angle_sum = 0.0
        dist_sum = 0.0

        for match in cluster[4]:
            (x1, y1) = cluster[0][match.queryIdx].pt
            (x2, y2) = cluster[2][match.trainIdx].pt

            angle = math.atan2(y2 - y1, x2 - x1)
            if angle < 0.0:
                angle = 2.0 * math.pi + angle
            angle_sum = angle_sum + angle

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            dist_sum = dist_sum + distance

        transforms.append((angle_sum / len(cluster[4]), dist_sum / len(cluster[4])))

    # for each pair of clusters
    for i in range(0, len(clustered_matches) - 1):
        for j in range(i + 1, len(clustered_matches)):
            if clustered_matches[i] is not None and clustered_matches[j] is not None:
                angle_i, dist_i = transforms[i]
                total_i = len(clustered_matches[i][4])

                angle_j, dist_j = transforms[j]
                total_j = len(clustered_matches[j][4])

                # if the transformations are equivalent
                if abs(angle_i - angle_j) <= angle_diff_tolerance and abs(dist_i - dist_j) <= dist_diff_tolerance:
                    # merges the current two clusters
                    offset1 = len(clustered_matches[i][0])
                    offset2 = len(clustered_matches[i][2])
                    for match in clustered_matches[j][4]:
                        match.queryIdx = match.queryIdx + offset1
                        match.trainIdx = match.trainIdx + offset2

                    clustered_matches[i][0].extend(clustered_matches[j][0])
                    clustered_matches[i][1] = np.vstack((clustered_matches[i][1], clustered_matches[j][1]))
                    clustered_matches[i][2].extend(clustered_matches[j][2])
                    clustered_matches[i][3] = np.vstack((clustered_matches[i][3], clustered_matches[j][3]))
                    clustered_matches[i][4].extend(clustered_matches[j][4])

                    # updates the resulting cluster's transformation data
                    angle_sum = angle_i * total_i + angle_j * total_j
                    dist_sum = dist_i * total_i + dist_j * total_j
                    transforms[i] = ((angle_sum / (total_i + total_j), dist_sum / (total_i + total_j)))

                    # cleans the old j-th cluster
                    clustered_matches[j] = None
                    transforms[j] = None

    # generates and returns the method output
    output = []
    for cluster in clustered_matches:
        if cluster is not None:
            output.append(cluster)
    return output

def _match_keypoints(image_shape1, keypoints1, descriptions1, image_shape2, keypoints2, descriptions2,
          min_match_count=3):
    # defines the two sets of keypoints to be matched
    """
    @ Joao comment
    If none keypoint was not provided return
    """
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return []

    # performs g2nn keypoint selection
    keypoints1, descriptions1, keypoints2, descriptions2 = _g2nn_keypoint_selection(keypoints1, descriptions1,
                                                                                    keypoints2, descriptions2)
    """
    @ Joao comment
    If none keypoint was found return
    """
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return []

    # clusterizes the remaining keypoints
    clusters1 = _clusterize_keypoints(keypoints1, descriptions1, image_shape1)
    clusters2 = _clusterize_keypoints(keypoints2, descriptions2, image_shape2)
    # for each pair of clusters
    clustered_matches = []
    for i in range(0, len(clusters1)):
        for j in range(0, len(clusters2)):
            # tries to find geometrically consistent matches
            keypoints_i = clusters1[i][0]
            descriptions_i = np.array(clusters1[i][1], dtype=np.float32)

            keypoints_j = clusters2[j][0]
            descriptions_j = np.array(clusters2[j][1], dtype=np.float32)

            """
            Acuna's Work
            If within a cluster, more than three keypoints are matched against the same cluster, then define those two clusters as matched clusters
            """
            if (len(keypoints_i)<= 3) or (len(keypoints_j)<= 3):
                continue
            
            ij_matches = _consistent_match(keypoints_i, descriptions_i, keypoints_j, descriptions_j)
            
            if len(ij_matches) >= min_match_count:

                clustered_matches.append([keypoints_i, descriptions_i, keypoints_j, descriptions_j, ij_matches])

    # conciliates similar clusters, regarding the estimated image-space transformations
    if len(clustered_matches):
        clustered_matches = _conciliate_clusters(clustered_matches)

    # returns the obtained clusters
    return clustered_matches

def match_keypoints(image_shape1, keypoints1, descriptions1, image_shape2, keypoints2, descriptions2,
          min_match_count=3):
    """
    Try match keypoings from img1 towards img2 and vice-versa
    The larger resulting cluster will be returned
    """
    
    try:
        clusters_img1_img2 = _match_keypoints(image_shape1, keypoints1, descriptions1, image_shape2, keypoints2, descriptions2,
          min_match_count=3)
    except cv2.error as e:
        clusters_img1_img2 = []
        print(e)
    except :
        raise
    
    try:
        clusters_img2_img1 = _match_keypoints(image_shape2, keypoints2, descriptions2, image_shape1, keypoints1, descriptions1,
          min_match_count=3)
    except cv2.error as e:
        clusters_img2_img1 = []
        print(e)
    except:
        raise
        
    return clusters_img1_img2 if len(clusters_img1_img2) >= len(clusters_img2_img1) else clusters_img2_img1


# Selects, from the given set of <matches>, the largest set of matches that are geometrically consistent
# with the <i>-th and <j>-th given matches. The matches are of the OpenCV DMatch type and therefore need
# the sets of keypoints <keypoints1> and <keypoints2> to compute consistency. The displacement threshold
# <displacement_thresh> is a tolerance, in pixels, to consider deviant matches still consistent.
# Returns the indices of the selected matches.
def __select_consistent_matches(matches, i, j, keypoints1, keypoints2, displacement_thresh=30):
    # selected matches
    selected_matches = []

    # if the given ij matches share keypoints, there is nothing to select
    if matches[i].queryIdx == matches[j].queryIdx or matches[i].trainIdx == matches[j].trainIdx:
        return selected_matches

    # query and train keypoints in matrix format
    query_points = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints1])
    train_points = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints2])

    # puts query and train images in the same scale
    query_i_point = query_points[matches[i].queryIdx]
    query_j_point = query_points[matches[j].queryIdx]
    query_ij_distance = math.sqrt(
        (query_i_point[0] - query_j_point[0]) ** 2 + (query_i_point[1] - query_j_point[1]) ** 2)

    train_i_point = train_points[matches[i].trainIdx]
    train_j_point = train_points[matches[j].trainIdx]
    train_ij_distance = math.sqrt(
        (train_i_point[0] - train_j_point[0]) ** 2 + (train_i_point[1] - train_j_point[1]) ** 2)

    # if the involved keypoints are too close to eah other, there is nothing to select
    if query_ij_distance == 0.0 or train_ij_distance == 0.0:
        return selected_matches

    distance_ratio = query_ij_distance / train_ij_distance

    if distance_ratio > 1.0:
        scale_matrix = np.zeros((3, 3))
        scale_matrix[0, 0] = distance_ratio
        scale_matrix[1, 1] = distance_ratio
        scale_matrix[2, 2] = 1.0
        train_points = cv2.perspectiveTransform(np.float32([train_points]), scale_matrix)[0]


    elif distance_ratio < 1.0:
        scale_matrix = np.zeros((3, 3))
        scale_matrix[0, 0] = 1.0 / distance_ratio
        scale_matrix[1, 1] = 1.0 / distance_ratio
        scale_matrix[2, 2] = 1.0
        query_points = cv2.perspectiveTransform(np.float32([query_points]), scale_matrix)[0]

    # computes and performs the rotation of the train image towards the query image
    query_i_point = query_points[matches[i].queryIdx]
    query_j_point = query_points[matches[j].queryIdx]
    query_angle = math.atan2(query_j_point[1] - query_i_point[1], query_j_point[0] - query_i_point[0])
    if query_angle < 0.0:
        query_angle = 2.0 * math.pi + query_angle

    train_i_point = train_points[matches[i].trainIdx]
    train_j_point = train_points[matches[j].trainIdx]
    train_angle = math.atan2(train_j_point[1] - train_i_point[1], train_j_point[0] - train_i_point[0])
    if train_angle < 0.0:
        train_angle = 2.0 * math.pi + train_angle

    train_angle_correction = query_angle - train_angle
    sine = math.sin(train_angle_correction)
    cosine = math.cos(train_angle_correction)

    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, 0] = cosine
    rotation_matrix[0, 1] = -sine
    rotation_matrix[1, 0] = sine
    rotation_matrix[1, 1] = cosine
    rotation_matrix[2, 2] = 1.0

    train_points = cv2.perspectiveTransform(np.float32([train_points]), rotation_matrix)[0]

    # computes and performs the translation of the train image towards the query image
    query_i_point = query_points[matches[i].queryIdx]
    train_i_point = train_points[matches[i].trainIdx]

    translation_matrix = np.zeros((3, 3))
    translation_matrix[0, 0] = 1.0
    translation_matrix[1, 1] = 1.0
    translation_matrix[2, 2] = 1.0
    translation_matrix[0, 2] = query_i_point[0] - train_i_point[0]
    translation_matrix[1, 2] = query_i_point[1] - train_i_point[1]

    train_points = cv2.perspectiveTransform(np.float32([train_points]), translation_matrix)[0]

    # selects the geometrically consistent matches
    used_query_idx = []
    used_train_idx = []

    for m in range(len(matches)):
        if matches[m].queryIdx in used_query_idx or matches[m].trainIdx in used_train_idx:
            continue

        query_x = query_points[matches[m].queryIdx][0]
        query_y = query_points[matches[m].queryIdx][1]

        train_x = train_points[matches[m].trainIdx][0]
        train_y = train_points[matches[m].trainIdx][1]

        if math.sqrt((query_x - train_x) ** 2 + (query_y - train_y) ** 2) < displacement_thresh:
            selected_matches.append(m)
            used_query_idx.append(matches[m].queryIdx)
            used_train_idx.append(matches[m].trainIdx)

    # returns the selected consistent matches
    return selected_matches
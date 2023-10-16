from re import I
import warnings

from typing import List, Union, Tuple, Optional

import cv2
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from PIL import Image

def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Performs G2NN selection over the given two sets of keypoints and their respective descriptions
# (<keypoints1>, <descriptions1>), (<keypoints2>, <descriptions2>). Please provide <within_same_image>
# as TRUE in the case of the two keypoint sets coming from the same (single) image. Parameter <k_rate>
# in [0.0, 1.0] helps to define how many neighbors are matched to each given keypoint. Parameter
# <nndr_threshold> in [0.0, 1.0] is the maximum value used to consider a match useful, according to its
# difference (distance-wise) to the next closest match (G2NN principle).
# Returns the indices of the selected keypoints, for each one of the given two sets.
def _g2nn_keypoint_selection(keypoints1, descriptions1, keypoints2, descriptions2,
                             k_rate=0.5, nndr_threshold=0.75, eps=1e-7,
                             matching_method='BF'):
    # defines the two sets of keypoints to be matched
    # (smaller set: keypoints1, larger set: keypoints2)
    swapped = False
    if len(keypoints2) < len(keypoints1):
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1
        swapped = True

    # matches keypoints1 towards keypoints2
    if matching_method == "BF":
        knn_matches = cv2.BFMatcher().knnMatch(descriptions1, descriptions2, k=max(1, int(round(len(keypoints1) * k_rate))))
    elif matching_method == "FLANN":
        #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        #knn_matches = matcher.knnMatch(descriptions1, descriptors2, 2)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        knn_matches = flann.knnMatch(descriptions1,descriptions2,k=2)

    # g2NN match selection
    selected_matches = []
    for _, matches in enumerate(knn_matches):
        for i in range(0, len(matches) - 1):
            if matches[i].distance / (matches[i + 1].distance + eps) < nndr_threshold:
                selected_matches.append(matches[i])
            else:
                break

    # keypoint and description selection, based on selected matches
    indices1 = []
    indices2 = []
    distances = []

    for match in selected_matches:

        if (match.queryIdx not in indices1) and (match.trainIdx not in indices2):
            indices1.append(match.queryIdx)
            indices2.append(match.trainIdx)
            distances.append(match.distance)

        else:
            """
            The following code was added by @Joao
            """
            # Check if keypoint was already included in the indices list
            # if not, check the distance of the matched desc
            # if the distance is less than the included one, overwrite
            # the match with the new one, otherwise keep the old match
            if match.queryIdx not in indices1:
                i = indices2.index(match.trainIdx)
            else:
                i = indices1.index(match.queryIdx)
            if distances[i] > match.distance:
                indices1[i] = match.queryIdx
                indices2[i] = match.trainIdx
                distances[i] = match.distance

    selected_keypoints_1 = [index for index in indices1]
    selected_keypoints_2 = [index for index in indices2]

    # undoes swapping, if it is the case
    if swapped:
        selected_keypoints_1, selected_keypoints_2 = selected_keypoints_2, selected_keypoints_1

    # returns the selected keypoints
    return selected_keypoints_1, selected_keypoints_2

# Selects, from the given set of <matches>, the largest set of matches that are geometrically consistent
# with the <i>-th and <j>-th given matches. The matches are of the OpenCV DMatch type and therefore need
# the sets of keypoints <keypoints1> and <keypoints2> to compute consistency. The displacement threshold
# <displacement_thresh> is a tolerance, in pixels, to consider deviant matches still consistent.
# Returns the indices of the selected matches.
def __select_consistent_matches(matches, i, j, keypoints1, keypoints2, displacement_thresh=15):
    # if the given ij matches share keypoints, there is nothing to select
    if matches[i].queryIdx == matches[j].queryIdx or matches[i].trainIdx == matches[j].trainIdx:
        return []

    # query and train keypoints in matrix format
    query_points = keypoints1
    train_points = keypoints2

    # puts query and train images in the same scale
    query_i_point = query_points[matches[i].queryIdx]
    query_j_point = query_points[matches[j].queryIdx]
    query_ij_distance = np.sqrt(
        (query_i_point[0] - query_j_point[0]) ** 2 + (query_i_point[1] - query_j_point[1]) ** 2)

    train_i_point = train_points[matches[i].trainIdx]
    train_j_point = train_points[matches[j].trainIdx]
    train_ij_distance = np.sqrt(
        (train_i_point[0] - train_j_point[0]) ** 2 + (train_i_point[1] - train_j_point[1]) ** 2)

    # if the involved keypoints are too close to eah other, there is nothing to select
    if query_ij_distance == 0.0 or train_ij_distance == 0.0:
        return []

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
    query_angle = np.arctan2(query_j_point[1] - query_i_point[1], query_j_point[0] - query_i_point[0])
    if query_angle < 0.0:
        query_angle = 2.0 * np.pi + query_angle

    train_i_point = train_points[matches[i].trainIdx]
    train_j_point = train_points[matches[j].trainIdx]
    train_angle = np.arctan2(train_j_point[1] - train_i_point[1], train_j_point[0] - train_i_point[0])
    if train_angle < 0.0:
        train_angle = 2.0 * np.pi + train_angle

    train_angle_correction = query_angle - train_angle
    sine = np.sin(train_angle_correction)
    cosine = np.cos(train_angle_correction)

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
    selected_matches = []

    used_query_idx = []
    used_train_idx = []

    used_query_pt = []
    used_train_pt = []

    for m in range(len(matches)):
        if matches[m].queryIdx in used_query_idx or matches[m].trainIdx in used_train_idx:
            continue

        query_x = query_points[matches[m].queryIdx][0]
        query_y = query_points[matches[m].queryIdx][1]

        train_x = train_points[matches[m].trainIdx][0]
        train_y = train_points[matches[m].trainIdx][1]

        if (query_x, query_y) in used_query_pt or (train_x, train_y) in used_train_pt:
            continue

        if np.sqrt((query_x - train_x) ** 2 + (query_y - train_y) ** 2) < displacement_thresh:
            selected_matches.append(m)

            used_query_idx.append(matches[m].queryIdx)
            used_train_idx.append(matches[m].trainIdx)

            used_query_pt.append((query_x, query_y))
            used_train_pt.append((train_x, train_y))

    # returns the selected consistent matches
    return selected_matches

# Computes, from the given two sets of keypoints and their respective descriptions (<keypoints1>, <descriptions1>),
# (<keypoints2>, <descriptions2>), the largest set of matches that are geometrically consistent among themselves.
# Parameter <nndr_thresh> in [0.0, 1.0] helps to select useful keypoints (2NN, a.k.a. Lowe's NNDR selection).
# Returns a list of matches of the OpenCV DMatch type.
def _consistent_match(keypoints1, descriptions1, keypoints2, descriptions2, nndr_thresh=0.85, eps=1e-7):
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
    for _, matches in enumerate(knn_matches):
        if len(matches) > 1 and matches[0].distance / (matches[1].distance + eps) < nndr_thresh:
            good_matches.append(matches[0])
    good_matches.sort(key=lambda m: m.distance)

    # selects the largest set of geometrically consistent matches
    selected_matches = []
    for i in range(0, len(good_matches) - 1):
        for j in range(i + 1, len(good_matches)):
            consistent_matches = __select_consistent_matches(good_matches, i, j, keypoints1, keypoints2)
            if len(consistent_matches) > len(selected_matches):
                selected_matches = consistent_matches[:]

    # prepares the selected matches to be returned
    answer = []
    for i in selected_matches:
        # fixes swap, if needed
        if swapped:
            good_matches[i].queryIdx, good_matches[i].trainIdx = good_matches[i].trainIdx, good_matches[i].queryIdx

        answer.append(good_matches[i])

    answer.sort(key=lambda m: m.distance)
    return answer

# Clusterizes the given <keypoints> according to their (x, y) positions within the source image.
# The shape (height, width) <image_shape> of the source image must also be given.
# Additional clustering configuration parameters:
# <conn_neighbor_rate> - Rate in [0.0, 1.0] to define the number of keypoints used to lock connectivity in the
#                        clustering solution;
#   <dist_thresh_rate> - Rate in [0.0, 1.0] to define the distance threshold used in the clustering solution;
#          <cpu_count> - Number of CPU cores used in clustering. Give -1 to use all cores.
# Returns the computed clusters as a list of lists containing the indices of the clustered keypoints.
def _clusterize_keypoints(keypoints, image_shape, conn_neighbor_rate=0.1, dist_thresh_rate=0.003, cpu_count=-1):
    if len(keypoints) == 0:
        return [[]]

    if len(keypoints) == 1:
        return [[0]]

    positions = keypoints

    nb_count = int(round(len(positions) * conn_neighbor_rate))
    dist_thresh = image_shape[0] * image_shape[1] * dist_thresh_rate

    if nb_count > 0:
        forced_conn = kneighbors_graph(X=positions, n_neighbors=nb_count, n_jobs=cpu_count)
        clustering = AgglomerativeClustering(n_clusters=None, connectivity=forced_conn, distance_threshold=dist_thresh)
    else:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh)

    clustering.fit(positions)
    labels = clustering.labels_

    clusters = {}
    for i in range(0, len(labels)):
        label = labels[i]

        if label not in clusters.keys():
            clusters[label] = []

        clusters[label].append(i)

    return [clusters[label] for label in clusters.keys()]


# Conciliates the given set of match clusters <clustered_matches>, based on their image-space transformation
# agreement. The image-space information is obtained through the given sets of keypoints <keypoints1> and
# <keypoints2>. Parameters <angle_diff_tolerance> (in radians) and <dist_diff_tolerance> (in pixels) define the
# tolerance to consider two clusters equivalent in terms of transformations.
# Returns the conciliated (merged) list of match clusters.
def _conciliate_clusters(clustered_matches, keypoints1, keypoints2,
                         angle_diff_tolerance=0.174533, dist_diff_tolerance=15, scale_estim_try=10):
    # defines the two sets of keypoints to be matched
    # (smaller set: keypoints1, larger set: keypoints2)
    swapped = False

    if len(keypoints2) < len(keypoints1):
        keypoints1, keypoints2 = keypoints2, keypoints1
        for match_cluster in clustered_matches:
            for m in match_cluster:
                m.queryIdx, m.trainIdx = m.trainIdx, m.queryIdx
        swapped = True

    # estimates the average transformation for each cluster
    transforms = []
    for match_cluster in clustered_matches:
        # estimates the difference in scale between the two matched clusters (one cluster in each image)
        scale_ratios = []
        for i in range(0, len(match_cluster) - 1):
            for j in range(i + 1, len(match_cluster)):
                # points and distance on image 1
                (x11, y11) = keypoints1[match_cluster[i].queryIdx]
                (x12, y12) = keypoints1[match_cluster[j].queryIdx]
                dist1 = np.sqrt((x11 - x12) ** 2 + (y11 - y12) ** 2)

                # points and distance on image 2
                (x21, y21) = keypoints2[match_cluster[i].trainIdx]
                (x22, y22) = keypoints2[match_cluster[j].trainIdx]
                dist2 = np.sqrt((x21 - x22) ** 2 + (y21 - y22) ** 2)

                # if the distances are not zero, estimates the difference in scale
                if dist1 > 0.0 and dist2 > 0.0:
                    scale_ratios.append(dist1 / dist2)

            if len(scale_ratios) > scale_estim_try:
                break

        if len(scale_ratios) > 0:
            scale_ratios = np.mean(scale_ratios)
        else:
            scale_ratios = 0.0

        # estimates the average angle and distance between the matched pairs of keypoints across images
        if scale_ratios > 0.0:
            angle_sum = 0.0
            dist_sum = 0.0

            for match in match_cluster:
                (x1, y1) = keypoints1[match.queryIdx]

                pt2 = keypoints2[match.trainIdx]
                (x2, y2) = (pt2[0] * scale_ratios, pt2[1] * scale_ratios)

                angle = np.arctan2(y2 - y1, x2 - x1)
                if angle < 0.0:
                    angle = 2.0 * np.pi + angle
                angle_sum = angle_sum + angle

                distance = np.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0)
                dist_sum = dist_sum + distance

            transforms.append((angle_sum / len(match_cluster), dist_sum / len(match_cluster)))

        else:
            transforms.append(None)

    # for each pair of clusters
    for i in range(0, len(clustered_matches) - 1):
        if transforms[i] is None:
            continue

        for j in range(i + 1, len(clustered_matches)):
            if transforms[j] is None:
                continue

            if clustered_matches[i] is not None and clustered_matches[j] is not None:
                angle_i, dist_i = transforms[i]
                total_i = len(clustered_matches[i])

                angle_j, dist_j = transforms[j]
                total_j = len(clustered_matches[j])

                # if the transformations are equivalent
                if abs(angle_i - angle_j) <= angle_diff_tolerance and abs(dist_i - dist_j) <= dist_diff_tolerance:
                    # merges the current two clusters
                    clustered_matches[i].extend(clustered_matches[j])

                    # updates the resulting cluster's transformation data
                    angle_sum = angle_i * total_i + angle_j * total_j
                    dist_sum = dist_i * total_i + dist_j * total_j
                    transforms[i] = (angle_sum / (total_i + total_j), dist_sum / (total_i + total_j))

                    # cleans the old j-th cluster
                    clustered_matches[j] = None
                    transforms[j] = None

    # generates and returns the method output
    output = []
    for match_cluster in clustered_matches:
        if match_cluster is not None:
            # fixes swap, if needed
            if swapped:
                for m in match_cluster:
                    m.queryIdx, m.trainIdx = m.trainIdx, m.queryIdx

            output.append(match_cluster)
    return output






# Approach 1 - Use the whole image
# Use RANSAC or other transformation
def _check_alignment(p1, p2, displacement_thresh=5):
    """
    check if p1 and p2 are near
    """
    if np.sqrt((p1[0]- p2[0]) ** 2 + (p1[1]- p2[1]) ** 2) < displacement_thresh:
        return True
    return False

def consistent_keypoints(keypoints1, keypoints2, alignment_method='CV_RANSAC'):
    """use the homography matrix to align the images and check their alignment"""

    check_alignment_consistency = False
    if alignment_method in ['CV_RANSAC', "CV_LMEDS", "CV_RHO"]:
        check_alignment_consistency = True
    if alignment_method == 'CV_RANSAC':
        (H, inliers) = cv2.findHomography(keypoints1, keypoints2, cv2.RANSAC, 0.5, 0.999, 100000)
    elif alignment_method == 'CV_LMEDS':
        (H, inliers) = cv2.findHomography(keypoints1, keypoints2, cv2.LMEDS, 0.5, 0.999, 100000)
    elif alignment_method == 'CV_RHO':
        (H, inliers) = cv2.findHomography(keypoints1, keypoints2, cv2.RHO, 0.5, 0.999, 100000)
    elif alignment_method == 'CV_DEGENSAC':
        H, inliers = cv2.findFundamentalMat(keypoints1, keypoints2, cv2.USAC_DEFAULT, 0.5, 0.999, 100000)
        consistent_keypoints_indices = inliers.flatten()
        consistent_keypoints_indices = consistent_keypoints_indices > 0
    elif alignment_method == 'CV_MAGSAC':
        H, inliers = cv2.findFundamentalMat(keypoints1, keypoints2, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        consistent_keypoints_indices = inliers.flatten()
        consistent_keypoints_indices = consistent_keypoints_indices > 0
    else:
        raise NotImplementedError

    # Discard Outliers
    #
    # There is no need to discard outliers
    # since the transformation and alignment
    # check will be performed in all kpts
    #####################################

    if check_alignment_consistency:
        pts = np.float32(keypoints1).reshape(-1,1,2)
        ## Get keypoints of the aligned image
        aligned_keypoints1 = cv2.perspectiveTransform(pts, H)
        aligned_keypoints1 = np.round(aligned_keypoints1, 1).astype(np.int32)
        aligned_keypoints1 = aligned_keypoints1.squeeze()

        consistent_keypoints_indices = []
        for i in range(len(keypoints1)):
            p1 = aligned_keypoints1[i]
            p2 = keypoints2[i]
            if _check_alignment(p1, p2):
                consistent_keypoints_indices.append(i)

    keypoints1 = keypoints1[consistent_keypoints_indices]
    keypoints2 = keypoints2[consistent_keypoints_indices]

    return keypoints1, keypoints2

def align_keypoints(
              keypoints1: np.array,
              desc1: np.array ,
              keypoints2: np.array ,
              desc2: np.array ,
              matching_method:str ="BF",
              kpts_threshold:int = 25,
              alignment_method = "CV_RANSAC",
            )-> Tuple[float,float]:
    """
    Use all keypoints from the img1 and img2 to geometric align their content (if possible).
    In other words, this funcion will find the transformation that better align the two sets of keypoints.
    Return the list of keypoints used to align both image or an empty list in case
    their aren't shared keypoints between the input images

    Args:
        keypoints1
        desc1
        keypoints2
        desc2
        kpts_threshold:
            number of keypoints to be considered a matche among both images
        return_area:
            return the percentage of content shared area of img1 and img2
    Return:
        matched_kpts_img1
        matched_kpts_img2
    """

    # Get descriptors/keypoints and their flipped version
    # Match the descriptors, note that we are using a higher threshold
    keypoints_indices1, keypoints_indices2 = _g2nn_keypoint_selection(keypoints1, desc1, keypoints2, desc2,
                             k_rate=0.5, nndr_threshold=0.85, eps=1e-7,
                             matching_method=matching_method)
    if len(keypoints_indices1) == 0 or len(keypoints_indices2)==0:
        return [], []

    keypoints1 = keypoints1[keypoints_indices1,: ]
    keypoints2 = keypoints2[keypoints_indices2, :]

    if len(keypoints1) < kpts_threshold:
            return [],[]

    # Check the consistency of the matched keypoints
    keypoints1, keypoints2 = consistent_keypoints(keypoints1,
                                                 keypoints2,
                                                 alignment_method)
    if len(keypoints1) < kpts_threshold:
            return [],[]

    return keypoints1, keypoints2

def shared_content_area(im, shared_keypoints):
    """
    Given an image (im) and the shared_keypoints of this image with another, calculate
    the content shared area between im and the other image.
    """
    hull_img = cv2.convexHull(np.array(shared_keypoints, "int32"))

    mask_hull = np.zeros((im.shape[0],im.shape[1]),"uint8")
    mask_hull= cv2.fillPoly(mask_hull, pts =[hull_img], color=1)
    shared_area = np.sum(mask_hull)/(mask_hull.shape[0]*mask_hull.shape[1])

    return shared_area

def homography_alignment_area(img1_dict, img2_dict, alignment_method="CV_RANSAC", matching_method='BF',
                                min_kpts_matches=9):
    """
    Match two images based on their shared content area

    img1_dict contains :
        image_path
        keypoints_path
        desc_path

    Return the content shared area from both images
    """

    kpts1 = np.load(img1_dict['keypoints_path'])
    kpts2 = np.load(img2_dict['keypoints_path'])

    desc1 = np.load(img1_dict['desc_path']).astype(np.float32)
    desc2 = np.load(img2_dict['desc_path']).astype(np.float32)

    im1 = np.array(Image.open(img1_dict['image_path'])).astype(np.uint)
    im2 = np.array(Image.open(img2_dict['image_path'])).astype(np.uint)


    selected_keypoints1, selected_keypoints2 = align_keypoints(
                                                                kpts1,
                                                                desc1,
                                                                kpts2,
                                                                desc2,
                                                                alignment_method=alignment_method,
                                                                matching_method=matching_method)

    if len(selected_keypoints1) < min_kpts_matches and len(selected_keypoints2) < min_kpts_matches:
        return 0,0

    return shared_content_area(im1, selected_keypoints1), shared_content_area(im2, selected_keypoints2)


def homography_alignment(img1_dict, img2_dict, alignment_method="CV_RANSAC",
                            matching_method="BF", debug_image=""):
    """
    Align two images based on their shared keypoints

    img1_dict contains :
        image_path
        keypoints_path
        desc_path

    Return the number of shared keypoints
    """

    kpts1 = np.load(img1_dict['keypoints_path'])
    kpts2 = np.load(img2_dict['keypoints_path'])

    desc1 = np.load(img1_dict['desc_path'])
    desc2 = np.load(img2_dict['desc_path'])


    selected_keypoints1, selected_keypoints2 = align_keypoints(
                                                                 kpts1,
                                                                 desc1,
                                                                 kpts2,
                                                                 desc2,
                                                                 alignment_method=alignment_method,
                                                                 matching_method=matching_method)

    if len(selected_keypoints1) == 0 or len(selected_keypoints2) == 0:
        return 0,0

    # DEBUG
    if debug_image:
        im1 = np.array(Image.open(img1_dict['image_path'])).astype(np.uint)
        im2 = np.array(Image.open(img2_dict['image_path'])).astype(np.uint)
        hull_debug = draw_hull(im1, selected_keypoints1, im2, selected_keypoints2)
        Image.fromarray(hull_debug).save(debug_image)

    return len(selected_keypoints1), len(selected_keypoints2)


def cluster_alignment(img1_dict,
                    img2_dict,
                    matching_method= "BF",
                    min_match_count=9,
                    debug_image=""
                    ):
    """
    Match two images using clusters of keypoints.
    The clusters are agglomeration based on the coordinates of matched keypoints


    Return
    matched_keypoints1, matched_keypoints2, cluster_matches

    Return the indices of the matched keypoints from image1 and image 2; and the keypoint clusters

    """

    kpts1 = np.load(img1_dict['keypoints_path'])
    kpts2 = np.load(img2_dict['keypoints_path'])

    desc1 = np.load(img1_dict['desc_path'])
    desc2 = np.load(img2_dict['desc_path'])

    im1 = np.array(Image.open(img1_dict['image_path'])).astype(np.uint)
    im2 = np.array(Image.open(img2_dict['image_path'])).astype(np.uint)

    keypoint_indices1, keypoint_indices2 = _g2nn_keypoint_selection(kpts1, desc1, kpts2, desc2,
                             matching_method=matching_method)

    selected_keypoints1 = kpts1[keypoint_indices1]
    selected_keypoints2 = kpts2[keypoint_indices2]


    clusters1 = _clusterize_keypoints(selected_keypoints1, im1.shape[:2])
    for c in clusters1:
        for i in range(len(c)):
            c[i] = keypoint_indices1[i]

    clusters2 = _clusterize_keypoints(selected_keypoints2, im2.shape[:2])
    for c in clusters2:
        for i in range(len(c)):
            c[i] = keypoint_indices2[i]

    # for each pair of clusters
    clustered_matches = []
    for i in range(0, len(clusters1)):
        for j in range(0, len(clusters2)):

            # tries to find geometrically consistent matches
            keypoints_i = [kpts1[k] for k in clusters1[i]]
            descriptions_i = np.array([desc1[k] for k in clusters1[i]], dtype=np.float32)

            keypoints_j = [kpts2[k] for k in clusters2[j]]
            descriptions_j = np.array([desc2[k] for k in clusters2[j]], dtype=np.float32)

            ij_matches = _consistent_match(keypoints_i, descriptions_i, keypoints_j, descriptions_j)
            for m in ij_matches:
                m.queryIdx = clusters1[i][m.queryIdx]
                m.trainIdx = clusters2[j][m.trainIdx]

            if len(ij_matches) >= min_match_count:
                clustered_matches.append(ij_matches)

    # conciliates similar clusters, regarding the estimated image-space transformations
    clustered_matches = _conciliate_clusters(clustered_matches, kpts1, kpts2)

    if len(clustered_matches) == 0:
        return [], []
    if len(clustered_matches[0]) ==0:
        return [], []

    # returns the obtained clusters
    keypoints_indices1 = [ i.queryIdx for cluster in clustered_matches for i in cluster]
    keypoints_indices2 = [ i.trainIdx for cluster in clustered_matches for i in cluster]

    # Selected keypoints are within the matched clusters

    selected_keypoints1 = kpts1[keypoints_indices1]
    selected_keypoints2 = kpts2[keypoints_indices2]

    # DEBUG
    if debug_image:
        im1 = np.array(Image.open(img1_dict['image_path'])).astype(np.uint)
        im2 = np.array(Image.open(img2_dict['image_path'])).astype(np.uint)
        hull_debug = draw_hull(im1, selected_keypoints1, im2, selected_keypoints2)
        Image.fromarray(hull_debug).save(debug_image)

    return selected_keypoints1, selected_keypoints2

def cluster_alignment_area(img1_dict,
                    img2_dict,
                    matching_method="BF",
                    min_match_count=9,
                    debug_image=""

                    ):
    """
    Match two images using clusters of keypoints.
    The clusters are agglomeration based on the coordinates of matched keypoints

    debug_image -> save an image with the convex hull of the mathching keypoints
    """

    im1 = np.array(Image.open(img1_dict['image_path'])).astype(np.uint)
    im2 = np.array(Image.open(img2_dict['image_path'])).astype(np.uint)

    # Get keypoints within each matched cluster
    selected_keypoints1 , selected_keypoints2 = cluster_alignment(img1_dict,
                                                                  img2_dict,
                                                                  min_match_count=min_match_count,
                                                                  matching_method=matching_method)
    # DEBUG
    if debug_image:
        im1 = np.array(Image.open(img1_dict['image_path'])).astype(np.uint)
        im2 = np.array(Image.open(img2_dict['image_path'])).astype(np.uint)
        hull_debug = draw_hull(im1, selected_keypoints1, im2, selected_keypoints2)
        Image.fromarray(hull_debug).save(debug_image)
    if len(selected_keypoints1)>0:
        shared_area =  shared_content_area(im1, selected_keypoints1), shared_content_area(im2, selected_keypoints2)
    else:
        shared_area = (0,0)
    return shared_area


def draw_hull(im1, keypoints1, im2, keypoints2):
    # Get Convex Hull
    hull_img1 = cv2.convexHull(np.array(keypoints1, "int32"))
    hull_img2 = cv2.convexHull(np.array(keypoints2, "int32"))

    draw_hull1 = cv2.drawContours(im1.copy().astype(np.uint8), [hull_img1], 0, 0, 2, 8)
    draw_hull2 = cv2.drawContours(im2.copy().astype(np.uint8), [hull_img2], 0, 0, 2, 8)

    # Get keypoints match on the original space
    matches_mask = []
    hull_points_img1 = []
    hull_points_img2 = []
    index = 0
    for p1, p2 in zip(hull_img1, hull_img2):

        x, y = p1[0].astype("float")
        point_img1 = cv2.KeyPoint(x,y,1)
        x, y = p2[0].astype("float")
        point_img2 = cv2.KeyPoint(x,y,1)
        hull_points_img1.append(point_img1)
        hull_points_img2.append(point_img2)
        d = cv2.DMatch()
        d.trainIdx = index
        d.queryIdx = index
        matches_mask.append(d)
        index+=1
        draw_params = dict(#matchColor = (0,255,0),
                    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    #borderColor=(255,255,255)
                    )
    return cv2.drawMatches(draw_hull1,hull_points_img1,draw_hull2,hull_points_img2,matches_mask,None,**draw_params)

#     ____                      __  ___     __      __ 
#    /  _/_ _  ___ ____ ____   /  |/  /__ _/ /_____/ / 
#   _/ //  ' \/ _ `/ _ `/ -_) / /|_/ / _ `/ __/ __/ _ \
#  /___/_/_/_/\_,_/\_, /\__/ /_/  /_/\_,_/\__/\__/_//_/
#                 /___/                                
#   _   ___               ___          __  _         
#  | | / (_)__ __ _____ _/ (_)__ ___ _/ /_(_)__  ___ 
#  | |/ / (_-</ // / _ `/ / /_ // _ `/ __/ / _ \/ _ \
#  |___/_/___/\_,_/\_,_/_/_//__/\_,_/\__/_/\___/_//_/
#                                                    
#  
#                                           Version 0.1
#  
#      JP. Cardenuto <phillipe.cardenuto@ic.unicamp.br>
#                                             Jun, 2021
#  
########################################################
#
# This file implements different forms to visualize an 
# image match
#
########################################################

import cv2
import numpy as np
import cv2
from PIL import Image
from typing import List, Union, Tuple, Optional
from numpy.linalg.linalg import LinAlgError

def angle_kps(p1: Tuple[int,int],
              p2: Tuple[int,int],
              width
) -> float:
    """
    Get the angle between  two points (relative to the origin(x=0,y=0)) from two image 'width' apart from each other
    ( as their were side-by-side).
    We consider that p1 belongs to the first image, that it is located in the origin, and p2 belongs to the second image
    that  is apart 'width' from the origin.

    Args: 
        p1: 
           Point of image 1 (with its (0,0) located at the origin of the system (x=0,y=0))
        p2: 
        Point of image 2 (with its (0,0) located at (x=width, y=0) of the system
    width: distance between image 1 and image 2.
        This parameters is used to calculate the angle between p1 (from img1)
        and p2 (from img2) considerating that they are side-by-side
        and img1 is on the left and img2 in on the right 
                    |  IMG1   |  IMG2  |
                    -  width  -
    Return:
        Angle in degrees in range [0,90]
    """
    
    x1, y1 = p1
    x2, y2 = p2

    angle = np.arctan2((y1-y2), (x1-(x2+width))) *180 / np.pi
    
    angle = angle%360
    if angle >= 180:
        angle -= 180
    if angle >= 90:
        angle = 180 - angle
        
    return angle


def content_match_keypoints(img1: np.ndarray,
                             img2: np.ndarray,
                             top_keypoints_matches: int = 200,
) -> bool:
    """
    Return True if the input share content, based on the
    number of their keypoints matching

    Args:
        img1: numpy.array
            image 1 that shares content with img2
        img2: numpy.array
            image 2 that shares content with img1
    """

    # Match Flags using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    knn_matches = flann.knnMatch(des1,des2,k=2)

    # Keep the top matches
    knn_matches = sorted(knn_matches, key=lambda x:x[0].distance)
    knn_matches = knn_matches[:top_keypoints_matches]
    
    if len(knn_matches) < 20:
        return 0
    # List of matching points 
    ptsA = np.zeros((len(knn_matches), 2), dtype="float")
    ptsB = np.zeros((len(knn_matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(knn_matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kp1[m[0].queryIdx].pt
        ptsB[i] = kp2[m[0].trainIdx].pt

    # use the homography matrix to align the images
    try:
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RHO)
        (h, w) = img2.shape[:2]
        aligned_img1 = cv2.warpPerspective(img1, H, (w, h))
    except cv2.error:
        return 0

    # Get keypoints of the aligned image
    kp_aligned, des_aligned = sift.detectAndCompute(aligned_img1,None)    
    knn_matches = flann.knnMatch(des_aligned,des2,k=2)

    # Draw only the good matches, so create a mask
    shared_keypoints = []
    for i,(m,n) in enumerate(knn_matches):
        #if m.distance < 0.7*n.distance:
        p1 = kp_aligned[m.queryIdx ].pt
        p2 = kp2[m.trainIdx ].pt
        angle = angle_kps(p1,p2,aligned_img1.shape[1])
        if angle < 1:
            shared_keypoints.append(p1)

    return len(shared_keypoints) if len(shared_keypoints) > 20 else 0

def highlight_shared_content(img1: np.ndarray,
                             img2: np.ndarray,
                             top_keypoints_matches: int = 200,
                             draw_hull: bool = False
) -> np.ndarray:
    """
    Return an image highlighting the shared content of img1 and img2
    after matching their keypoints using SIFT.

    There is two operation mode for this method:
    1. Highlight the shared content area on both images, by illuminance.
    2. Drawing the convex hull of the shared contend for both images, drawing a line
        for each respective convex hull vertice.

    Args:
        img1: numpy.array
            image 1 that shares content with img2
        img2: numpy.array
            image 2 that shares content with img1
        top_keypoints_matches: int
            Maximum number of keypoints to create the conves hull
    draw_hull: boolean
        If False: operates as mode 1
        Otherwise: operates as mode 2

    Return:
        result: numpy.array
         an image with both (img1 and img2) concatenated side-by-side with their
         shared area higlighted according to the chosen operation mode.
    """

    # Match Flags using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    knn_matches = flann.knnMatch(des1,des2,k=2)

    # Keep the top matches
    knn_matches = sorted(knn_matches, key=lambda x:x[0].distance)
    knn_matches = knn_matches[:top_keypoints_matches]

    # List of matching points 
    ptsA = np.zeros((len(knn_matches), 2), dtype="float")
    ptsB = np.zeros((len(knn_matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(knn_matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kp1[m[0].queryIdx].pt
        ptsB[i] = kp2[m[0].trainIdx].pt

    # use the homography matrix to align the images
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    (h, w) = img2.shape[:2]
    aligned_img1 = cv2.warpPerspective(img1, H, (w, h))

    # Get keypoints of the aligned image
    kp_aligned, des_aligned = sift.detectAndCompute(aligned_img1,None)    
    knn_matches = flann.knnMatch(des_aligned,des2,k=2)

    # Draw only the good matches, so create a mask
    shared_keypoints_p1 = []
    shared_keypoints_p2 = []
    for i,(m,n) in enumerate(knn_matches):
        #if m.distance < 0.7*n.distance:
        p1 = kp_aligned[m.queryIdx ].pt
        p2 = kp2[m.trainIdx ].pt
        angle = angle_kps(p1,p2,aligned_img1.shape[1])
        if angle < 1:
            shared_keypoints_p1.append(p1)
            shared_keypoints_p2.append(p2)

    if len(shared_keypoints_p1) < 10:
        raise RuntimeError(f"Images do not share enough content only {len(shared_keypoints_p1)} keypoint(s) matched.")


    if draw_hull:
        # Get Convex Hull
        hull_img2 = cv2.convexHull(np.array(shared_keypoints_p2, "int32"))
        hull_img1 = hull_img2.copy().astype("float32")
        hull_img1 = cv2.perspectiveTransform(hull_img1, np.linalg.inv(H)).astype("int") 

        draw_hull1 = cv2.drawContours(img1.copy(), [hull_img1], 0, 0, 2, 8)
        draw_hull2 = cv2.drawContours(img2.copy(), [hull_img2], 0, 0, 2, 8)

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
   
    
    # Convex Hull keypoints img 1, making the inverse of H (homography) based on the shared_keypoits
    hull_img2 = cv2.convexHull(np.array(shared_keypoints_p2, "int32"))
    hull_img1 = hull_img2.copy().astype("float32")
    hull_img1 = cv2.perspectiveTransform(hull_img1, np.linalg.inv(H)).astype("int") 
   
    # Mask of hull1
    mask_hull1 = np.zeros((img1.shape[0],img1.shape[1],3),"uint8")
    mask_hull1= cv2.fillPoly(mask_hull1, pts =[hull_img1], color=(255,255,255))
    mask_hull1= cv2.blur(mask_hull1, (29,29))
    
    # Create highlighted img1
    mask_hull1 = Image.fromarray(mask_hull1).convert("RGBA")
    img1_pil = Image.fromarray(img1).convert("RGBA")
    img1 = np.array(Image.blend(mask_hull1, img1_pil, 0.6).convert("RGB"))
    
    # Convex Hull Keypoints img 2
    # Mask of hull1
    mask_hull2 = np.zeros((img2.shape[0],img2.shape[1],3),"uint8")
    mask_hull2 = cv2.fillPoly(mask_hull2, pts =[hull_img2], color=(255,255,255))
    mask_hull2 = cv2.blur(mask_hull2, (29,29))

    # Create highlighted img2
    mask_hull2 = Image.fromarray(mask_hull2).convert("RGBA")
    img2_pil = Image.fromarray(img2).convert("RGBA")
    img2 =  np.array(Image.blend(mask_hull2, img2_pil, 0.6).convert("RGB"))

    max_height = max(img1.shape[0], img2.shape[0])
    vis_img = np.zeros((max_height , img1.shape[1] + img2.shape[1]+1,3),dtype='uint8')
    vis_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    vis_img[0:img2.shape[0], img1.shape[1]+1:] = img2
    return vis_img


def align_images(img1: np.ndarray,
                 img2: np.ndarray,
                 top_keypoints_matches: int = 200,
                 show_keypoints: bool = False,
                 visualization_keypoints: int = 100
) -> Tuple[np.ndarray, np.ndarray] :
    """
    Align img1 with img2 considering that they share some content.
    During the alignment process, this function extract keypoints of
    both images using SIFT and match them using knn.

    After the alignment the function returns a visualization of the alignmnet, and 
    the homography transformation H that aligned img1 towards img2.

    Args:
        img1:cv2.image 
        img2: cv2.image
        top_keypoints_matches: 
            Maximum number of keypoints matches used during alignment
        show_keypoints:
            If True shows the matching keypoints in the final visualization
        visualization_keypoints:
            Number of Mathching keypoints in the final visualization
    Return
        visualization_img:
            Image with the homgraphic transformation of img1 (that aligns with img2) side-by-side with img2
        homography_transformation:
            Matrix containing the homographic transformation applied to image 1 to align it with img2.

    """

    # Match Flags using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    knn_matches = flann.knnMatch(des1,des2,k=2)

    # Keep the top matches
    knn_matches = sorted(knn_matches, key=lambda x:x[0].distance)
    knn_matches = knn_matches[:top_keypoints_matches]

    

    # List of matching points 
    ptsA = np.zeros((len(knn_matches), 2), dtype="float")
    ptsB = np.zeros((len(knn_matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(knn_matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kp1[m[0].queryIdx].pt
        ptsB[i] = kp2[m[0].trainIdx].pt

    # use the homography matrix to align the images
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    (h, w) = img2.shape[:2]
    aligned_img1 = cv2.warpPerspective(img1, H, (w, h), borderValue=(255,255,255))

   
    # Get keypoints of the aligned image
    kp_aligned, des_aligned = sift.detectAndCompute(aligned_img1,None)    
    knn_matches = flann.knnMatch(des_aligned,des2,k=2)

    # Draw only the good matches, so create a mask
    alignedMask = [[0,0] for i in range(len(knn_matches))]
    count_vis_keypoints = 0
    for i,(m,n) in enumerate(knn_matches):
        #if m.distance < 0.7*n.distance:
        p1 = kp_aligned[m.queryIdx ].pt
        p2 = kp2[m.trainIdx ].pt
        angle = angle_kps(p1,p2,aligned_img1.shape[1])
        if angle < 1:
            count_vis_keypoints +=1
            alignedMask[i]=[1,0]
        # Limite the visualization keypoints
        if count_vis_keypoints == visualization_keypoints :
            break

    if count_vis_keypoints < 10:
        raise RuntimeError(f"Images do not share enough content only {count_vis_keypoints} keypoint(s) matched.")
    # If show_keypoints is false, just return the aligned image
    # Else Generate a keypoints visualization using 
    # the aligned image and img2
    if show_keypoints == False:
        max_height = max(aligned_img1.shape[0], img2.shape[0])
        vis_img = np.zeros((max_height , aligned_img1.shape[1] + img2.shape[1]+1,3),dtype='uint8')
        vis_img[0:aligned_img1.shape[0], 0:aligned_img1.shape[1]] = aligned_img1
        vis_img[0:img2.shape[0], aligned_img1.shape[1]+1:] = img2
        return vis_img, H


    draw_params = dict(matchColor = (0,255,0),
                    matchesMask = alignedMask,
                    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_img = cv2.drawMatchesKnn(aligned_img1,kp_aligned,img2,kp2,knn_matches,None,**draw_params)
    return vis_img, H

    
def shared_content_area(img1: np.ndarray, img2: np.ndarray, top_keypoints_matches=1000) -> Tuple[float,float]:
    """
    Calculates the percentage area of img1 that are shared with img2
    Args:
        img1: numpy.array
            image 1 that shares content with img2
        img2: numpy.array
            image 2 that shares content with img1
    Return:
        shared_area_img1:
            Percentage of img1 shared with img2
        shared_area_img2:
            Percentage of img2 shared with img1
    """
    # Match Flags using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    knn_matches = flann.knnMatch(des1,des2,k=2)

    # Keep the top matches
    knn_matches = sorted(knn_matches, key=lambda x:x[0].distance)
    knn_matches = knn_matches[:top_keypoints_matches]    
    # knn_matches = [x for x in knn_matches if x[0].distance > 0.7*x[1].distance]

    # List of matching points 
    ptsA = np.zeros((len(knn_matches), 2), dtype="float")
    ptsB = np.zeros((len(knn_matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(knn_matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kp1[m[0].queryIdx].pt
        ptsB[i] = kp2[m[0].trainIdx].pt

    # use the homography matrix to align the images
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    (h, w) = img2.shape[:2]
    aligned_img1 = cv2.warpPerspective(img1, H, (w, h))

    # Get keypoints of the aligned image
    kp_aligned, des_aligned = sift.detectAndCompute(aligned_img1,None)    
    knn_matches = flann.knnMatch(des_aligned,des2,k=2)

    # Draw only the good matches, so create a mask
    shared_keypoints_p1 = []
    shared_keypoints_p2 = []
    for i,(m,n) in enumerate(knn_matches):
        #if m.distance < 0.7*n.distance:
        p1 = kp_aligned[m.queryIdx ].pt
        p2 = kp2[m.trainIdx ].pt
        angle = angle_kps(p1,p2,aligned_img1.shape[1])
        if angle < 1:
            shared_keypoints_p1.append(p1)
            shared_keypoints_p2.append(p2)

    if len(shared_keypoints_p1) < 25:
        return 0, 0
        #raise RuntimeError(f"Images do not share enough content only {len(shared_keypoints_p1)} keypoint(s) matched.")

    # Convex Hull keypoints img 1, making the inverse of H (homography) based on the shared_keypoits
    hull_img2 = cv2.convexHull(np.array(shared_keypoints_p2, "int32"))
    hull_img1 = hull_img2.copy().astype("float32")
    try:
        hull_img1 = cv2.perspectiveTransform(hull_img1, np.linalg.inv(H)).astype("int") 
    except LinAlgError:
        return 0,0
   
    # Mask of hull1
    mask_hull1 = np.zeros((img1.shape[0],img1.shape[1]),"uint8")
    mask_hull1= cv2.fillPoly(mask_hull1, pts =[hull_img1], color=1)
    area_shared_img1 = np.sum(mask_hull1)/(mask_hull1.shape[0]*mask_hull1.shape[1])
    
    # Convex Hull Keypoints img 2
    # Mask of hull2
    mask_hull2 = np.zeros((img2.shape[0],img2.shape[1]),"uint8")
    mask_hull2 = cv2.fillPoly(mask_hull2, pts =[hull_img2], color=(1))
    area_shared_img2= np.sum(mask_hull2)/(mask_hull2.shape[0]*mask_hull2.shape[1])

    return  area_shared_img1, area_shared_img2
from typing import List, Tuple
import cyvlfeat
import cv2
import numpy as np
from PIL import Image
from src.sila_image_descriptor import sift_detect_rsift_describe # sila method

from skimage.filters import gaussian

# To speedup everything, test isolate modules, and let it running if ok

CLAHE_APPLIER = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def vlfeat_sift(input_image,
                 peak_thresh=0.1,
                 edge_thresh=12,
                 hist_equalization = True,
                 eps=1e-7,
                 root_sift = True,
                 flip = False

                ) -> Tuple[List, List]:

    """
    Using vlfeat library generates sift descriptors
    Parameters:
        input_image: path to image
    
    Return:
        Keypoints List(x,y) -- shape = (N, 2) , N number of keypoints found
        Descriptors List(feats) -- shape = (N, 128)
    """

    # To use sift:
    """
    `image`` must be
    ``float32`` and greyscale (either a single channel as the last axis, or no
    channel)
    """

    # Read image
    im = np.array(Image.open(input_image)).astype(np.float64)
    if flip:
        im = cv2.flip(im, 1)

    # Convert to gray
    if len(im.shape) > 2:
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np
           .float64)

    # applies CLAHE histogram equalization over the given image
    if hist_equalization:
        im = CLAHE_APPLIER.apply(im.astype(np.uint8)).astype(np.float64)

    # Create descriptor
    locations, descriptors = cyvlfeat.sift.sift( 
                                    im,
				                    edge_thresh = edge_thresh,
					                peak_thresh = peak_thresh,
					                compute_descriptor=True)

    # 'locations' returns as [Y,X,S,TH] getting only the position
    # Change format to [X,Y]
    locations = locations[:, [0,1]]
    locations[:, [1,0]] = locations[:, [0,1]]

    # locations to int
    locations = np.round(locations,1).astype(int)

    if root_sift:
        descriptors = descriptors/ (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors= np.sqrt(descriptors)
    return locations, descriptors



def vlfeat_dsift(input_image,
                 flip = False
                 ) -> Tuple[List, List]:
    """
    https://www.vlfeat.org/overview/dsift.html

    Does NOT compute a Gaussian scale space of the image. Instead, the image                                           
    should be pre-smoothed at the desired scale level.     
    One could use ``skimage.filters.gaussian`` to smooth the image.

    COMMENTS:
    This method extracts sift features for all points 4x4 bbx from the image.
    Therefore, it should be used with a post-processing method to eliminate
    not relevant descriptors

    the ``dsift`` descriptors cover the bounding box
    specified by ``bounds = [YMIN, XMIN, YMAX, XMAX]``. Thus the top-left bin
    of the top-left descriptor is placed at ``(YMIN, XMIN)``. The next
    three bins to the right are at ``YMIN + size``, ``YMIN + 2*size``,
    ``YMIN + 3*size``. The Y coordinate of the center of the first descriptor is
    therefore at ``(YMIN + YMIN + 3*size) / 2 = YMIN + 3/2 * size``. For
    instance, if ``YMIN=1`` and ``size=3`` (default values), the Y
    coordinate of the center of the first descriptor is at
    ``1 + 3/2 * 3 = 5.5``. For the second descriptor immediately to its right
    this is ``5.5 + step``, and so on.

    Documentation @ https://github.com/menpo/cyvlfeat/blob/master/cyvlfeat/sift/dsift.py

    """

    # Read image
    im = np.array(Image.open(input_image)).astype(np.float64)

    if flip:
        im = cv2.flip(im, 1)

    # Convert to gray
    if len(im.shape) > 2:
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np
           .float64)
       
    binSize = 8
    magnif = 3
    im = gaussian(im, np.sqrt((binSize/magnif)**2 - .25))   

    # Create descriptor
    locations, descriptors = cyvlfeat.sift.dsift( 
                                    im,
                                    size = binSize,
					                float_descriptors=True)

    # 'locations' returns as [Y,X,S,TH] getting only the position
    # Change format to [X,Y]
    locations = locations[:, [0,1]]
    locations[:, [1,0]] = locations[:, [0,1]]

    # locations to int
    locations = np.round(locations,1).astype(int)

    return locations, descriptors


def vlfeat_phow(input_image, flip = False):
    """
    https://www.vlfeat.org/matlab/vl_phow.html

    https://www.vlfeat.org/overview/dsift.html#tut.dsift.phow
    """

    # Read image
    im = np.array(Image.open(input_image)).astype(np.float64)
    if flip:
        im = cv2.flip(im, 1)

    # Convert to gray
    if len(im.shape) > 2:
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np
           .float64)
       
    # Create descriptor
    locations, descriptors = cyvlfeat.sift.phow( 
                                    im,
					                )

    # 'locations' returns as [Y,X,S,TH] getting only the position
    # Change format to [X,Y]
    locations = locations[:, [0,1]]
    locations[:, [1,0]] = locations[:, [0,1]]

    # locations to int
    locations = np.round(locations,1).astype(int)

    return locations, descriptors


# Keeps doubling the size of the given <image> and respective <mask> up to the point of having
# more pixels than <min_image_size>.
# Returns the re-sized image and mask, and the number of times they were re-sized.
def _increase_image_if_necessary(image, mask=None, min_image_size=10000):
    resize_count = 0

    while image.shape[0] * image.shape[1] < min_image_size:
        image = cv2.resize(image, (0, 0), fx=2.0, fy=2.0)

        if mask is not None:
            mask = cv2.resize(mask, (0, 0), fx=2.0, fy=2.0)

        resize_count = resize_count + 1

    return image, mask, resize_count

def cv_sift(input_image,
            kp_count=2000,
            hist_equalization = True,
            contrastThreshold = 0.0,
            sigma = 3.2,
            eps=1e-7,
            root_sift = True,
            flip = False

            ) -> Tuple[List, List]:
    """
    Using cv2 library generates sift descriptors
    Parameters:
        input_image: path to image
    
    Return:
        Keypoints List(x,y) -- shape = (N, 2) , N number of keypoints found
        Descriptors List(feats) -- shape = (N, 128)
    """


    im = np.array(Image.open(input_image)).astype(np.uint8)
    if flip:
        im = cv2.flip(im, 1)
    im, _, resize_count = _increase_image_if_necessary(im)

    # Convert to gray
    if len(im.shape) > 2:
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    if hist_equalization:
        # applies CLAHE histogram equalization over the given image
        im = CLAHE_APPLIER.apply(im)

    # SIFT object
    sift_detector = cv2.SIFT_create(nfeatures=kp_count,
                                    contrastThreshold=contrastThreshold,
                                    sigma=sigma)

    # detects SIFT keypoints over the given image
    keypoints = sift_detector.detect(im)


    # if no keypoints were detected, returns empty keypoints and descriptions
    if len(keypoints) == 0:
        return np.array([]), np.array([])

    # sorts the obtained keypoints according to their response, and keeps only the top-<kp_count> ones
    keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)
    del keypoints[kp_count:]

    # describes the remaining keypoints
    keypoints, descriptions = sift_detector.compute(im, keypoints)
    descriptions = descriptions / (descriptions.sum(axis=1, keepdims=True) + eps)
    descriptions = np.sqrt(descriptions)


    # re-adjusts the obtained keypoints according to the change of image size
    if resize_count > 0:
        for kp in keypoints:
            kp.pt = (kp.pt[0] / (2.0 ** resize_count), kp.pt[1] / (2.0 ** resize_count))
            kp.size = kp.size / (2.0 ** resize_count)


    keypoints = np.array([ (int(np.round(k.pt[0],1)), int(np.round(k.pt[1],1)))
                        for k in keypoints])

    if root_sift:
        descriptions = descriptions / (descriptions.sum(axis=1, keepdims=True) + eps)
        descriptions = np.sqrt(descriptions)


    return keypoints, descriptions


def sila_method(input_image,
                kp_count = 2000,
                flip = False):
    """
    Using SILA descriptor 
    Parameters:
        input_image: path to image
    
    Return:
        Keypoints List(x,y) -- shape = (N, 2) , N number of keypoints found
        Descriptors List(feats) -- shape = (N, 128)
    """

    # Read image
    im = np.array(Image.open(input_image)).astype(np.float64)
    if flip:
        im = cv2.flip(im, 1)

    # Convert to gray
    if len(im.shape) > 2:
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    keypoints, descriptions, _ = sift_detect_rsift_describe(im, kp_count=kp_count, mask=None, mask_text=False, 
                                                            mask_background=False)

    keypoints = np.array([ (int(np.round(k.pt[0],1)), int(np.round(k.pt[1],1)))
                        for k in keypoints])

    return keypoints, descriptions
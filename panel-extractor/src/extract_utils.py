import sys
from pathlib import Path
import io
from PIL import Image

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# FILE = Path(__file__).absolute()
# sys.path.append(FILE.parents[0].as_posix())

from utils.general import check_img_size,  non_max_suppression, scale_coords, xyxy2xywh,xywh2xyxy, increment_path, save_one_box, clip_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device
from utils.augmentations import letterbox




def has_intersection(a, b, img_shape, threshold=0.4):
    """
    Verify if the inputs regions has any intersection
    Parameters
    ----------
    a: tuple
        bounding box coordinates (x0,y0,x1,y1) of region A
    b: tuple
        bounding box coordinates (x0,y0,x1,y1) of region B
    Return
    ------
    Boolean
        Result
    """
    img_width, img_height =  img_shape[0:2]
    a_img = np.zeros((img_height, img_width), bool)
    b_img = np.zeros((img_height, img_width), bool)
    inter = np.zeros((img_height, img_width), bool)
    ax0, ay0, ax1, ay1 = [int(round(i)) for i in a]
    bx0, by0, bx1, by1 = [int(round(i)) for i in b]

    a_img[ay0:ay1, ax0:ax1] = 1
    b_img[by0:by1, bx0:bx1] = 1

    a_area = a_img.sum()
    b_area = b_img.sum()

    if a_area ==0 or b_area ==0:
        return False

    inter = a_img * b_img
    inter_area = inter.sum()

    if inter_area / a_area > threshold or  \
        inter_area / b_area > threshold:
        return True
    return False

def union(a, b):
    """
    Union of 2 activated regions
    Parameters
    ----------
    a: tuple
        bounding box coordinates (x0,y0,x1,y1) of region A
    b: tuple
        bounding box coordinates (x0,y0,x1,y1) of region B
    Return
    ------
    x,y,w,h (tuple)
        (x,y,width,height) of the union region
     """
    x0 = int(min(a[0], b[0]))
    y0 = int(min(a[1], b[1]))
    x1 = int(max(a[2], b[2]))
    y1 = int(max(a[3], b[3]))
    min_conf = min(a[4], b[4])
    cls = a[5]
    return x0, y0, x1, y1, min_conf, cls

def join_overlapping_figures(coords, img_shape, threshold=0.4):
    """
    Loop over all activated areas the join the ones which overlap exceed
    the their area by a threshold
    """
    coords_cc = coords.copy()

    def join():
        i = 0
        while i < len(coords_cc):
            # Get i-th subimage coordinate
            *xyxy_i, _, cls_i = coords_cc[i]

            # Check if the i-th subimage has overlap with the rest of the images
            for j in range(i + 1, len(coords_cc)):
                *xyxy_j, _, cls_j = coords_cc[j]
                # Insert the coordinates on a BoundBox format

                # Check intersection between a and b and share the same class
                # If true, join the subimages, and discart one of them
                # --We are considering that the prediction was done using
                # an agnostic approach--.
                if has_intersection(xyxy_i, xyxy_j, img_shape, threshold=threshold) \
                   and  cls_i == cls_j:
                    coords_cc[i] = union(coords_cc[i], coords_cc[j])
                    coords_cc.pop(j)
                    i = -1
                    break

            i += 1


    join()
    return coords_cc


def save_one_box(xyxy, im, file='image.png', gain=1., pad=0, square=False, BGR=False, save=True):
    """
    source: https://github.com/ultralytics/yolov5
    Save a croped box from an the input image 'im' located at 'xyxy'

    Parameters
    ----------
    xyxy: tuple
        Box coordinate (x0,y0,x1,y1)
    im: image input

    file: save file name

    """
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save and (min(crop.shape[:2]) > 10):
        savepath = str(increment_path(file, mkdir=True).with_suffix('.png'))
        cv2.imwrite(savepath, crop)
    return crop

def preprocess_img(img, imgsz, stride, device):
    """
    source: https://github.com/ultralytics/yolov5
    """
    # Padded resize
    img = letterbox(img, imgsz, stride=stride)[0]

    # Adjust Image to Model
    img = img.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
    # img = img[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    return img

def nms_join(pred, img, timg, conf_thres=0.4, iou_thres=0.4, area_threshold=0.4):
    """
    Perform non max suppression join the bbox that have the same class and respect the
    thresholds.
    """
    # NMS and join overlap regions

    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=True, )
    for i, det in enumerate(pred):
         det[:, :4] = scale_coords(timg.shape[2:], det[:, :4], img.shape).round()

    pred = [ torch.tensor(join_overlapping_figures(
                            p.tolist(), img.shape, threshold=area_threshold))
                            for p in pred]

    return pred

def crop_figures(pred, img, classes, save_path, debug=False, save_img=False, save_txt=False):
    """
    source: https://github.com/ultralytics/yolov5
    pred: predicted bboxs
    img: np.array image
    classes: list of str, name of the each classes following the predict labels

    Return list of panels found
    [(panel_id, label, x0, y0, x1, y1)]
    """
    crop_index = 1
    debug_img = img.copy()
    panels = []
    for det in pred:  # detections per image
        if len(det):
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for *xyxy, conf, cls in reversed(det):

                # Add panel to panel list
                x0, y0, x1, y1 = xyxy
                panels.append((crop_index, classes[int(cls)], x0.item(), y0.item(), x1.item(), y1.item()))

                if save_img:
                    save_one_box(xyxy, img, file= f'{save_path}_{crop_index}_{classes[int(cls)]}.png', BGR=False)

                crop_index+=1

                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf)
                if save_txt:
                    with open(f'{save_path}.txt','a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if debug:
                    print(('%g ' * len(line)).rstrip() % line)
                    label = f'{classes[int(cls)]}-{conf:.2f}'
                    debug_img = plot_one_box(xyxy, debug_img, label=label, color=colors(0, False))

    if debug:
        return debug_img, panels

    return panels


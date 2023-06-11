#   ______ _                           
#  |  ____(_)                          
#  | |__   _  __ _ _   _ _ __ ___  ___ 
#  |  __| | |/ _` | | | | '__/ _ \/ __|
#  | |    | | (_| | |_| | | |  __/\__ \
#  |_|    |_|\__, |\__,_|_|  \___||___/
#             __/ |                    
#            |___/                     
#   ______      _                  _   _             
#  |  ____|    | |                | | (_)            
#  | |__  __  _| |_ _ __ __ _  ___| |_ _  ___  _ __  
#  |  __| \ \/ / __| '__/ _` |/ __| __| |/ _ \| '_ \ 
#  | |____ >  <| |_| | | (_| | (__| |_| | (_) | | | |
#  |______/_/\_\\__|_|  \__,_|\___|\__|_|\___/|_| |_|
#                                                    
#                                                    
#  
#                                         Version 0.1
#  
#    JP. Cardenuto <phillipe.cardenuto@ic.unicamp.br>
#                                           Sep, 2021
#  
#  ==================================================

import argparse
import sys
from pathlib import Path
import io
from PIL import Image

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path("/app/src/").absolute()
#FILE = Path("src/").absolute()
sys.path.append(FILE.as_posix())
from models.experimental import attempt_load
from utils.torch_utils import select_device
from extract_utils import *
from tqdm import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='model_4_class.pt', help='model.pt path')
    parser.add_argument('--input-path', '-i',required=True, nargs='+',type=str, help='path/to/figure. You can input multiple figures at the same time')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--save_img', type=bool, default=True, help='Save Extracted Images')
    parser.add_argument('--save-txt', type=bool, default=False, help="Save Extracted Images' BBox")
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output-path','-o', required=True, type=str, help='Directory path where the extracted figures will be saved')
    opt = parser.parse_args()
    return opt


def run(
    input_path,
    output_path,
    weights = '/app/model_4_class.pt',
    device = 'cpu',
    imgsz = 640,
    conf_thres=0.4,  # confidence threshold
    iou_thres=0.4,  # NMS IOU threshold
    save_txt=False,
    save_img=False
):
  
    # Read Model
    device = select_device(device)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Extract Each PDF From The Input
    input_path.sort()
    status_bar = tqdm(input_path, desc='Extraction', leave=True)
    
    csv_file = []
    for fig in status_bar:
        
        # Read PDF
        fig = Path(fig).absolute()  # to Path
            
        status_bar.set_description(f"Processing - {fig.stem}")
        status_bar.refresh() # to show immediately the update

        # Include Fig Name in save Path
        extract_path = str(Path(output_path) / fig.stem)
        
        # Open Image
        try:
            img = np.array(Image.open(fig).convert("RGB"))
        except:
            print(f"Warninig: Could not open figure '{fig.name}'")
            continue
        
        # If image is too small just pass
        if min(img.shape[:2]) < 30:
            continue

        # Torch Image
        timg = preprocess_img(img, imgsz,stride, device)

        # predict
        pred = model(timg)[0]
        # Apply NMS and Join Overlapping figs
        pred = nms_join(pred, img, timg, conf_thres=conf_thres,iou_thres=iou_thres )
        # Save Extracted Figures
        panels = crop_figures(pred, img, model.names,  extract_path, save_txt=save_txt, save_img=save_img)

        for panel in panels:
            csv_row = [str(fig.stem)] + [ str(i) for i in panel]
            csv_file.append(csv_row)
    

    with open(f"{output_path}/PANELS.csv", "w") as res_file:

        res_file.write("FIGNAME,ID,LABEL,X0,Y0,X1,Y1\n")

        for row in csv_file:
            row = ", ".join(list(row))
            res_file.write(f"{row}\n") 


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))













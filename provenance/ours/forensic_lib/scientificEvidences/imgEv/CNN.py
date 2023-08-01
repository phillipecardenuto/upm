"""
Use a ImageNet Pretrained model to generate a global Embedding for an Image
"""


"""
Image Embedding

Possible Models
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)
"""



from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import torch, torchvision
from PIL import Image

from forensic_lib.utils.img_utils import load_image, check_ext
from forensic_lib.utils.vector_utils import normalize_vector


from pathlib import PurePath
from typing import List, Union, Tuple, Optional

import numpy as np
from PIL import Image, ExifTags

from tqdm import tqdm

def load_default_image_model_or_preprocess():
    """
    model : object = None,
    preproces : object = None
    )-> Union[torch.nn.modules.container.Sequential, torchvision.transforms.transforms.Compose]:
    
    Load the default model=mobilnetV3 and preprocess operation= (Resize, Crop, Normalize)
    TODO Make this function a general model loading function
    
    """
    
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load MobileNet_V3 Model
    model = models.mobilenet_v3_large(pretrained=True)
    # Remove Top Layer (Classification)
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    
    return model, preprocess


class ImageDataset(Dataset):
    """
    TODO: Define valid extensions
    """
   
    def __init__(self,
                 image_files : Union[PurePath, str],
                 image_ids : List[int],
                 transform: torchvision.transforms
    ):
        
        # Remoe invalid image_files
        entry =  [(img_file, img_id) for img_file, img_id in 
                                     zip(image_files, image_ids)
                                     if check_ext(img_file)]
        
        self.image_files = [e[0] for e in entry ]
        self.image_ids = [e[1] for e in entry ]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        image = load_image(img_file)
        img_id = self.image_ids[idx]
        if self.transform:
            image = self.transform(image)
        return image, img_id
    
def get_image_embedding(
                        image_files: List[Union[PurePath, str]],
                        image_ids: List[int],
                        model : torch.nn.modules.container.Sequential,
                        transform : torchvision.transforms.transforms.Compose,
                        normalize: bool = False, 
                        **kwargs
)-> Tuple[List[List[float]], List[int]]:  
    
    use_gpu = kwargs.get('use_gpu') if kwargs.get('use_gpu') else False
    
    if torch.cuda.is_available() and use_gpu: 
        dev = "cuda:"+ str(kwargs.get('gpu_id')) if kwargs.get('gpu_id') else 'cuda:0'
    else: 
        dev = "cpu" 
    
    # No need to create overhead using batch
    if len(image_files) == 1:
        if check_ext(image_files[0]):
            img = transform(load_image(image_files[0])).unsqueeze(0) 
            
            with torch.no_grad():
                embedded_vectors = model(img).squeeze().numpy()
                
            if normalize:
                embedded_vectors = normalize_vector(embedded_vectors)
            return [list(embedded_vectors)], image_ids
    
    # Need to Create batch of images
    else:
        embedded_vectors = []
        ids = []
        img_dataset = ImageDataset(image_files, image_ids, transform)
        
        if use_gpu:    
            print("\rLoading Model to Device\r",end="")
            model = model.to(dev)
            print("Model on GPU              ",flush=True)

        batch_size = kwargs.get('batch_size') if kwargs.get('batch_size') else 64
        num_workers = kwargs.get('num_workers') if kwargs.get('num_workers') else 4
        img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        for imgs, ids_batch in tqdm(img_dataloader, total=np.ceil(len(img_dataset) / batch_size).astype(int)):
            imgs = imgs.to(dev)
            
            with torch.no_grad():
                output = model(imgs)
                output = output.squeeze().cpu().numpy()
                
            if normalize:
                output = normalize_vector(output)
                
            embedded_vectors += output.tolist()
            ids += [int(_id) for _id in ids_batch]
            
        if use_gpu:    
            print("\rRemove Model from Device\r",end="")
            model = model.to('cpu')
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
            print("Model on CPU             ",flush=True)
            
        return embedded_vectors, ids   
    

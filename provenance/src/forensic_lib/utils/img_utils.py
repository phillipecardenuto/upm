
from pathlib import PurePath
from typing import List, Union, Tuple

import numpy as np
from PIL import Image

IMG_FORMATS = ['JPG','JPEG', 'PNG', 'BMP', 'MPO', 'PPM', 'TIFF', 'GIF']
#logger = return_logger(__name__)

"""
We Might use this function to create the nodes in the Visualization Graph
"""
def preprocess_image(
    image, target_size: Tuple[int, int] = None, grayscale: bool = False
) -> np.ndarray:
    """
    Take as input an image as numpy array or Pillow format. Returns an array version of optionally resized and grayed
    image.
    Args:
        image: numpy array or a pillow image.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.
    Returns:
        A numpy array of the processed image.
        
    Original Method from https://github.com/idealo/imagededup/blob/3465540cc5c8fdf9254aff76069e28641dfc515f/imagededup/utils/image_utils.py
    """
    if isinstance(image, np.ndarray):
        image = image.astype('uint8')
        image_pil = Image.fromarray(image)

    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise ValueError('Input is expected to be a numpy array or a pillow object!')

    if target_size:
        image_pil = image_pil.resize(target_size, Image.ANTIALIAS)

    if grayscale:
        image_pil = image_pil.convert('L')

    return np.array(image_pil).astype('uint8')

def check_ext(
    image_file: Union[PurePath, str],
    img_formats: List[str] = IMG_FORMATS,
) -> bool:
    """
    Check if image extension name is valid according the img_formats argument.
    Return a boolean value according the extension of the image_file
    Args:
        image_file: Path to the image file.
        img_formats: List of allowed image formats.
    """
    
    image_extension = image_file.split('.')[-1].upper()
    return image_extension in img_formats

def load_image(
    image_file: Union[PurePath, str],
    target_size: Tuple[int, int] = None,
    grayscale: bool = False,
    img_formats: List[str] = IMG_FORMATS,
) -> np.ndarray:
    """
    Load an image given its path. Returns an array version of optionally resized and grayed image. Only allows images
    of types described by img_formats argument.
    Args:
        image_file: Path to the image file.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.
        img_formats: List of allowed image formats that can be loaded.
        
    Original Method from https://github.com/idealo/imagededup/blob/3465540cc5c8fdf9254aff76069e28641dfc515f/imagededup/utils/image_utils.py
    """
    try:
        img = Image.open(image_file)

        # validate image format
        if img.format not in img_formats:
            #logger.warning(f'Invalid image format {img.format}!')
            return None

        else:
            if img.mode != 'RGB':
                # convert to RGBA first to avoid warning
                # we ignore alpha channel if available
                img = img.convert('RGBA').convert('RGB')

            return img

    except Exception as e:
        #logger.warning(f'Invalid image file {image_file}:\n{e}')
        return None
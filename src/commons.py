
import logging

from pathlib import Path
from typing import List, Tuple

import cv2

import numpy
import numpy.typing

from data.bounding_box import BoundingBox

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def filextension(filepath: str) -> Tuple[bool, bool]:
    """
    returns: (
        is_image,
        is_video
    )
    """
    
    path = Path(filepath)
    return (
        path.suffix.lower() in IMAGE_EXTS,
        path.suffix.lower() in VIDEO_EXTS
    )
    
def fileslist(sample_path: str) -> List[str]:
    path = Path(sample_path)
    
    if (path.is_file()):
        return [str(sample_path)]
    
    elif (path.is_dir()):
        return [str(file) for file in path.glob("*")]
    
    return []
    

def read_video(video_path: str) -> cv2.VideoCapture:
    path = Path(video_path)
    
    if (path.is_file() and path.suffix.lower() in VIDEO_EXTS):
        return cv2.VideoCapture(str(path))
                                
    raise TypeError("path is not video file")

def read_image(image_path: str) -> cv2.typing.MatLike:
    path = Path(image_path)
    
    if(path.is_file() and path.suffix.lower() in IMAGE_EXTS):
        return cv2.imread(str(path))
    
    raise TypeError("path is not image file")

def read_images(image_path: str) -> List[cv2.typing.MatLike]:
    path = Path(image_path)
    
    results = []
    if (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
        results.append(read_image(str(path)))
        
    elif(path.is_dir()):
        for file in path.glob("*"):
            results.append(read_image(str(file)))
        
    return results

def get_ratio(
    originwidth: int,
    originheight: int,
    modelwidth: int,
    modelheight: int,
) -> float:
    return min(modelwidth / originwidth, modelheight / originheight)

def get_unpad_image(
    image: cv2.typing.MatLike,
    width: int,
    height: int,
) -> cv2.typing.MatLike:
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )
    
def get_blank_image(
    modelwidth: int,
    modelheight: int,
) -> numpy.typing.NDArray:
    return numpy.full(
        (modelheight, modelwidth, 3),
        (144, 144, 144),
        dtype=numpy.uint8,
    )
    
def get_offset_length(
    shape_width: int,
    shape_height: int,
    model_width: int,
    model_height: int,
    ratio: float = 0
):
    
    ratio = (
        get_ratio(
            shape_width,
            shape_height,
            model_width,
            model_height
        ) 
        if ratio == 0
        else ratio
    )
    
    unpad_width: int = int(round(shape_width * ratio))
    unpad_height: int = int(round(shape_height * ratio))
    
    offset_width: int = (model_width - unpad_width) // 2
    offset_height: int =  (model_height - unpad_height) // 2
    
    return (
        offset_width,
        offset_height
    )
    
def drawbox(
    image: cv2.typing.MatLike,
    box: BoundingBox,
) -> cv2.typing.MatLike:
    
    cv2.rectangle(
        image,
        box.lefttop(),
        box.rightbottom(),
        (0, 255, 255),
        2
    )
    
    return image


def drawlabel(
    image: cv2.typing.MatLike,
    box: BoundingBox,
) -> cv2.typing.MatLike:
    
    shape_height, shape_width = image.shape[:2]
    label = f"{str(box.id)}:{box.name}({int(box.confidence * 100)}%)"
    
    (labelw, labelh), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )
    
    left, top = box.lefttop()
    labelx = left
    labely = top - 10 if top - 10 > labelh else top + 10
    
    if (labelx + labelw > shape_width):
        labelx = shape_width - labelw
        
    if (labely + labelh > shape_height):
        labely = shape_height - labelh
        
    if (labelx - labelw < 0):
        labelx = 10
        
    if (labely - labelh < 0):
        labely = 10
        
    
    cv2.rectangle(
        image,
        (int(labelx), int(labely - labelh)),
        (int(labelx + labelw), int(labely + labelh)),
        (0, 255, 0),
        cv2.FILLED
    )
    
    cv2.putText(
        image,
        label,
        (labelx, labely + (labelh // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )
    
    return image

def preprocess_image(
    image: cv2.typing.MatLike,
    model_width: int,
    model_height: int,
) -> cv2.typing.MatLike:
    
    shape_height, shape_width = image.shape[:2]
    
    new_ratio: float = get_ratio(
        shape_width,
        shape_height,
        model_width,
        model_height
    )
    
    unpad_width: int = int(round(shape_width * new_ratio))
    unpad_height: int = int(round(shape_height * new_ratio))
        
    offset_width = (model_width - unpad_width) // 2
    offset_height =  (model_height - unpad_height) // 2
    
    # resize the origin image source
    unpad_image = get_unpad_image(
        image,
        unpad_width,
        unpad_height
    )
        
#     logger.debug(f"""
# shape: {shape_width}x{shape_height}
# ratio: {new_ratio}
# unpad: {unpad_width}x{unpad_height}
# unpad_image: {unpad_image.shape}
# model: {model_width}x{model_height}
#     """)
    
    # create a blank canvas for padding
    padded_image = get_blank_image(
        model_width,
        model_height
    )
    
    # replace the resized image into blank canvas for padding
    padded_image[
        offset_height:offset_height + unpad_height,
        offset_width:offset_width + unpad_width
    ] = unpad_image
    
    return padded_image


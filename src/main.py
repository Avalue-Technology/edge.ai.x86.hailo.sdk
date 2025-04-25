

import argparse
from pathlib import Path
import sys
import logging

import cv2
from data.model_information import ModelInformation
from model.hailort import Hailort
from model.onnxrt import Onnxrt
from model.runntime import Runtime

import commons
from model.tflitert import Tflitert

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)10s %(levelname)5s] %(filename)s.%(funcName)s:%(lineno)d: %(message)s"
)

logger = logging.getLogger(__name__)
windowname = "test"

parser = argparse.ArgumentParser()
parser.add_argument("--sample-path", type=str, help="sample path")
parser.add_argument("--model-path", type=str, help="model path")
parser.add_argument("--confidence", type=int, default=50, help="confidence threshold")
parser.add_argument("--threshold", type=int, default=50, help="nms filter threshold")
args = parser.parse_args()

sample_path  = args.sample_path # "/home/avalue/hailosdk/samples/images"
sample_files = commons.fileslist(sample_path)

model_path = args.model_path #"/home/avalue/hailosdk/models/object-detection/yolo/onnx/yolo11x.onnx"

confidence = args.confidence
threshold = args.threshold

cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_KEEPRATIO)

def loadonnx(onnxpath: str) -> Runtime:
    return Onnxrt(onnxpath)

def loadhef(hefpath: str) -> Runtime:
    return Hailort(hefpath)

def loadtflite(tflitepath: str) -> Runtime:
    return Tflitert(tflitepath)

def loadmodel(model_path: str) -> Runtime:
    mp = Path(model_path)

    if (mp.suffix.lower() == ".onnx"):
        return loadonnx(model_path)
        
    elif (mp.suffix.lower() == ".hef"):
        return loadhef(model_path)
    
    elif (mp.suffix.lower() == ".tflite"):
        return loadtflite(model_path)
    
    raise ValueError(f"unsupport model type: {mp.suffix}")

def main():
    
    runtime = loadmodel(model_path)

    index = 0
    max = len(sample_files)
    
    while True:
        sample_file = sample_files[index]
        
        is_image, is_video = commons.filextension(sample_file)
    
        if (is_image):
            display_inference_image(runtime, sample_file)
    
        elif (is_video):
            display_inference_video(runtime, sample_file)
    
        else:
            logger.error(f"sample_file: {sample_file} both not image or video")
            
        key = cv2.waitKeyEx(0)
        
        if key == ord('q') or key == ord('Q'):
            break

        elif key == 65361:  # Left Arrow
            index = (index - 1 + max) % max
            
        elif key == 65363:  # Right Arrow
            index = (index + 1) % max
            
    
    cv2.destroyAllWindows()

def display_inference_image(runtime: Runtime, filepath: str) -> None:
    logger.debug(filepath)
    
    input_image = commons.read_image(filepath)
    
    result = runtime.inference(
        input_image,
        confidence,
        threshold
    )
    
    cv2.imshow(windowname, result.image)
    


def display_inference_video(runtime: Runtime, filepath: str) -> None:
    capture = commons.read_video(filepath)
    
    while capture.isOpened():
        ret, frame = capture.read()
        if (not ret):
            break
        
        result = runtime.inference(
            frame,
            confidence,
            threshold
        )
        
        cv2.imshow(windowname, result.image)
        
        logger.info(f"spendtime: {result.spendtime}")
        
        key = cv2.waitKeyEx(1)
        if key == ord('q') or key == ord('Q'):
            break

if __name__ == "__main__":
    main()
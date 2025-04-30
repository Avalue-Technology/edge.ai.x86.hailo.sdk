
import argparse
import sys
import logging
import threading

from pathlib import Path
import time
from typing import Callable, Dict, List, Optional

import cv2

from vidgear.gears import CamGear

import commons

from data.inference_result import InferenceResult
from data.model_information import ModelInformation
from model.hailort import Hailort
from model.onnxrt import Onnxrt
from model.runntime import Runtime
from model.tflitert import Tflitert
from monitor import Monitor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)10s %(levelname)5s] %(filename)s.%(funcName)s:%(lineno)d: %(message)s"
)

logger = logging.getLogger(__name__)
windowname = "test"

parser = argparse.ArgumentParser()
parser.add_argument("-spath", "--sample-path", type=str, default=None, help="sample path")
parser.add_argument("-smjpeg", "--sample-mjpeg", type=str, default=None, help="cctv mjpeg url")
parser.add_argument("-syoutube", "--sample-youtube", type=str, default=None, help="youtube url")
parser.add_argument("-m", "--model-path", type=str, help="model path")
parser.add_argument("-c", "--confidence", type=int, default=50, help="confidence threshold")
parser.add_argument("-t", "--threshold", type=int, default=50, help="nms filter threshold")
parser.add_argument("-d", "--display", action="store_true", help="display inference results")
parser.add_argument("-l", "--loop", action="store_true", help="loop forever when input sample is video")
parser.add_argument("-f", "--fps", action="store_true", help="monitor inference frame per second when input sample is video")

args = parser.parse_args()

sample_path  = args.sample_path # "/home/avalue/hailosdk/samples/images"
sample_mjpeg = args.sample_mjpeg
sample_youtube = args.sample_youtube

model_path = args.model_path #"/home/avalue/hailosdk/models/object-detection/yolo/onnx/yolo11x.onnx"

confidence = args.confidence
threshold = args.threshold

is_display = args.display
is_loop = args.loop
is_monitor = args.fps

logger.debug(f"is_display: {is_display}, is_loop: {is_loop}, is_monitor: {is_monitor}")
logger.debug(f"samples: {sample_path} {sample_mjpeg} {sample_youtube}")

monitor_label_scale = 0.5
monitor_label_thickness = 1

monitor = Monitor(Path(model_path).name)

def loadonnx(onnxpath: str) -> Runtime:
    return Onnxrt(onnxpath, is_display)

def loadhef(hefpath: str) -> Runtime:
    return Hailort(hefpath, is_display)

def loadtflite(tflitepath: str) -> Runtime:
    return Tflitert(tflitepath, is_display)

def loadmodel(model_path: str) -> Runtime:
    mp = Path(model_path)

    if (mp.suffix.lower() == ".onnx"):
        return loadonnx(model_path)
        
    elif (mp.suffix.lower() == ".hef"):
        return loadhef(model_path)
    
    elif (mp.suffix.lower() == ".tflite"):
        return loadtflite(model_path)
    
    raise ValueError(f"unsupport model type: {mp.suffix}")


def drawmodelname(image: cv2.typing.MatLike, name: str) -> None:
    image_height, image_width = image.shape[:2]
    label: str = f"Model: {name}"
    
    (labelw, labelh), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        commons.get_label_scale(image_width),
        monitor_label_thickness,
    )
    
    labelx = (image_width // 2) - (labelw // 2)
    labely = image_height - labelh
    
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
        commons.get_label_scale(image_width),
        (0, 0, 0),
        monitor_label_thickness,
        cv2.LINE_AA
    )
    
    

def drawspendtime(image: cv2.typing.MatLike, spendtime: float):
    image_height, image_width = image.shape[:2]
    
    label: str = f"Latency: {int(spendtime * 1000)}ms"
    
    (labelw, labelh), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        commons.get_label_scale(image_width),
        monitor_label_thickness,
    )
    
    labelx = 0
    labely = image_height - labelh
    
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
        commons.get_label_scale(image_width),
        (0, 0, 0),
        monitor_label_thickness,
        cv2.LINE_AA
    )

def drawfps(image: cv2.typing.MatLike, framecount: float):
    image_height, image_width = image.shape[:2]
    label: str = f"FPS: {framecount:.1f}"
    
    (labelw, labelh), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        commons.get_label_scale(image_width),
        monitor_label_thickness
    )
    
    labelx = image_width - labelw
    labely = image_height - labelh
    
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
        commons.get_label_scale(image_width),
        (0, 0, 0),
        monitor_label_thickness,
        cv2.LINE_AA
    )
    

def main():
    if (is_display):
        cv2.namedWindow(
            windowname,
            cv2.WINDOW_NORMAL
        )
        
        cv2.setWindowProperty(
            windowname,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL
        )
        
    if (is_monitor):
        monitor.start()
        
    runtime = loadmodel(model_path)
    
    if (sample_path is not None):
        path = Path(sample_path)
        if(path.is_file() or path.is_dir()):
            sample_files = commons.fileslist(sample_path)
            display_inference_files(runtime, sample_files)
            
    elif (sample_mjpeg is not None):
        if(sample_mjpeg.find("http") >= 0):
            display_inference_url_mjpeg(runtime, sample_mjpeg)
        
    elif (sample_youtube is not None):
        if(sample_youtube.find("youtu") >= 0):
            display_inference_url_youtube(runtime, sample_youtube)
        
    if (is_display):
        cv2.destroyAllWindows()
        
def display_inference_files(runtime: Runtime, sample_files: List[str]):
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
        
        if (is_video):
            if (is_loop):
                key = cv2.waitKeyEx(1)
                
            else:
                key = cv2.waitKeyEx(0)
                
        else:
            key = cv2.waitKeyEx(0)
        
        if key == ord('q') or key == ord('Q'):
            break

        elif key == 65361:  # Left Arrow
            index = (index - 1 + max) % max
            
        elif key == 65363:  # Right Arrow
            index = (index + 1) % max
            
def display_inference_image(runtime: Runtime, filepath: str) -> None:
    logger.debug(filepath)
    
    input_image = commons.read_image(filepath)
    
    result = runtime.inference(
        input_image,
        confidence,
        threshold
    )
    
    if (is_display):
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
        
        if (is_monitor):
            monitor.add_count()
            monitor.add_spendtime(result.spendtime)
            drawmodelname(result.image, runtime.information.name)
            drawspendtime(result.image, monitor.spandtime)
            drawfps(result.image, monitor.framecount)
        
        if (is_display):
            cv2.imshow(windowname, result.image)
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                break

def display_inference_url_mjpeg(runtime: Runtime, url: str) -> None:
    capture = commons.read_url_video(url)
    
    if (not capture.isOpened()):
        logger.error(f"open url failed url: {url}")
        return None
    
    while(True):
        
        ret, frame = capture.read()
        if (not ret):
            capture = commons.read_url_video(url)
            continue
        
        result = runtime.inference(
            frame,
            confidence,
            threshold
        )
        
        if (is_monitor):
            monitor.add_count()
            monitor.add_spendtime(result.spendtime)
            drawmodelname(result.image, runtime.information.name)
            drawspendtime(result.image, monitor.spandtime)
            drawfps(result.image, monitor.framecount)
        
        if (is_display):
            cv2.imshow(windowname, result.image)
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                break
            
def display_inference_url_youtube(runtime: Runtime, url: str) -> None:
    options: Dict[str, str] = {
        # "CAP_PROP_FRAME_WIDTH": 1920, # resolution 320x240
        # "CAP_PROP_FRAME_HEIGHT": 1080,
        # "CAP_PROP_FPS": 60, # framerate 60fps
        "STREAM_RESOLUTION": "1080P",
    }
    
    
    stream = CamGear(
        source=url, # type: ignore
        stream_mode=True,
        logging=True,
        **options, # type: ignore
        
    ).start()
    
    while True:
    
        
        frame = stream.read()
        if (frame is None):
            break
        
        result = runtime.inference(
            frame,
            confidence,
            threshold
        )
        
        if (is_monitor):
            monitor.add_count()
            monitor.add_spendtime(result.spendtime)
            drawmodelname(result.image, runtime.information.name)
            drawspendtime(result.image, monitor.spandtime)
            drawfps(result.image, monitor.framecount)
        
        if (is_display):
            cv2.imshow(windowname, result.image)
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                break
    

if __name__ == "__main__":
    main()
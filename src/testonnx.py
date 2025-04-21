
from pathlib import Path
import sys
import logging

import cv2
import onnx

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)10s %(levelname)5s] %(filename)s.%(funcName)s:%(lineno)d: %(message)s"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

logger = logging.getLogger(__name__)

model: str = sys.argv[1]
samples_path: str = sys.argv[2]
confidence: int = int(sys.argv[3])
threshold: int = int(sys.argv[4])

samples = list(Path(samples_path).glob(f"*.*"))
samples_index = 0
samples_max = len(samples)

logger.debug(samples)

cv2.namedWindow("main", cv2.WINDOW_NORMAL)

rt = onnx.Onnxrt(model)

def inference_image(sample_path: str):
    image = cv2.imread(sample_path)
    
    result = rt.inference_image(
            image,
            confidence,
            threshold
        )
        
    cv2.imshow("main", result.image)
    logger.debug(f"spendtime: {result.spendtime}")
    

def inference_video(sample_path: str):
    capture = cv2.VideoCapture(sample_path)
    
    if not capture.isOpened():
        logger.error("open video failed")
        return None
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        result = rt.inference_image(
            frame,
            confidence,
            threshold
        )
        
        logger.debug(f"spendtime: {result.spendtime}")
        
        cv2.imshow("main", result.image)
        
        key = cv2.waitKeyEx(1)
        if key == ord('q') or key == ord('Q'):
            break

while True:
    
    sample_path = samples[samples_index]
    sample_type = sample_path.suffix.lower()
    
    if (sample_type in IMAGE_EXTS):
        inference_image(sample_path)
        
    elif(sample_type in VIDEO_EXTS):
        inference_video(sample_path)

    key = cv2.waitKeyEx(0)
    if key == ord('q') or key == ord('Q'):
        break

    elif key == 65361:  # Left Arrow
        samples_index = (samples_index - 1 + samples_max) % samples_max
        
    elif key == 65363:  # Right Arrow
        samples_index = (samples_index + 1) % samples_max
        
cv2.destroyAllWindows()


import numpy
import logging
import time

from typing import Dict
from collections import defaultdict

import cv2
import psutil

import tflite_runtime.interpreter as tflite

from data.inference_result import InferenceResult

logger = logging.getLogger(__name__)

class Tflite():
    
    def __init__(
		self,
		tflite_path: str,
	):
        
        self._tflite_path = tflite_path
        self._interpreter = tflite.Interpreter(model_path=tflite)
        self._interpreter.allocate_tensors()
        
        self._input_details = self._interpreter.get_input_details()
        self._input = self._input_details[0]
        
        self._batch, self._channels, self._height, self._width = (
			self._input.get("shape")
		)
        
        self._type = self._input.get("dtype")
        
        
    def inference_image(self, image_path: str, score: int) -> InferenceResult:
        
        source: cv2.typing.MatLike = cv2.imread(image_path)
        
        image: cv2.typing.MatLike = cv2.resize(source, (self._width, self._height))
        
        
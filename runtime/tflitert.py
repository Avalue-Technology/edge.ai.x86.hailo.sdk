
import logging
import platform
import time

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from collections import defaultdict

import cv2
import numpy
import numpy.typing
import psutil
import tflite_runtime.interpreter as tflite

from ..commons import utils
from ..commons.monitor import Monitor

from ..data.bounding_box import BoundingBox
from ..data.coco_90 import find_class_id
from ..data.inference_result import InferenceResult
from ..data.inference_source import InferenceSource
from ..data.model_information import ModelInformation

from .runtime import Runtime

logger = logging.getLogger(__name__)

class Tflitert(Runtime):

    def __init__(
        self,
        tflite_path: str,
    ):
        super().__init__()
        
        self._tflite_path = tflite_path
        
        self._session = tflite.Interpreter(model_path=tflite_path)
        self._session.allocate_tensors()
        
        self._input = self._session.get_input_details()[0]
        self._output = self._session.get_output_details()[0]
        
        logger.debug(self._input)
        logger.debug(self._output)
        
        self._batch_size, self._width, self._height, self._channels = (
            self._input.get("shape", (0, 0, 0, 0))
        )
        
        self._information = ModelInformation(
            Path(tflite_path).name,
            f"CPU {platform.processor()}",
            self._width,
            self._height
        )

    @property
    def temperature(self) -> int:
        coretemp = psutil.sensors_temperatures().get("coretemp")
        
        if (coretemp is not None):
            return int(coretemp[0].current)
        
        return 0
    
    def decode6(self, source: cv2.typing.MatLike, outputs: List):
        shape_height, shape_width = source.shape[:2]
        
        ratio = utils.get_ratio(
            shape_width,
            shape_height,
            self._width,
            self._height,
        )
        
        offset_width, offset_height = utils.get_offset_length(
            shape_width,
            shape_height,
            self._width,
            self._height,
            ratio
        )
        
        # x, y, w, h = outputs[0:4]
        x1, y1, x2, y2 = outputs[0:4]
        confidence = outputs[4]
        id = outputs[5]
        
        x1 = (x1 - offset_width) / ratio
        y1 = (y1 - offset_height) / ratio
        x2 = (x2 - offset_width) / ratio
        y2 = (y2 - offset_height) / ratio
        
        width = x2 - x1
        height = y2 - y1
        center = (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )
        
        box = BoundingBox()
        box.id = int(id) - 1
        box.name = find_class_id(None, box.id)
        box.confidence = confidence
        
        box.center = center
        box.width = width
        box.height = height
        
        return box
        
    
    def preprocess(
        self,
        source: cv2.typing.MatLike,
    ) -> numpy.typing.NDArray:
        
        image = utils.reshape(
            source,
            self._width,
            self._height
        )
        
        # data = numpy.array(image) / 255.0
        # data = numpy.transpose(data, (2, 0, 1))
        data = numpy.expand_dims(image, axis=0)
        
        return data.astype(numpy.uint8)
    
    
    def inference(self, source: InferenceSource) -> InferenceResult:
        input_data = self.preprocess(source.image)
        
        now = time.time()
        self._session.set_tensor(self._input.get("index"), input_data)
        self._session.invoke()
        end = time.time()
        spendtime = end - now
        
        if (not self.display):
            return InferenceResult(source)
        
        output_data = self._session.get_tensor(self._output.get("index"))
        output_data = numpy.squeeze(output_data)
        
        rows: int = self._output.get("shape", (1, 100, 7))[1] # _, rows, (_, y1, x1, y2, x2, score, id)
        
        boxes = []
        
        for i in range(rows):
            outputs = output_data[i]
            _, y1, x1, y2, x2, score, id = outputs
            
            if score >= (source.confidence / 100):
                box = self.decode6(
                    source.image,
                    [x1, y1, x2, y2, score, id]
                )
                
                boxes.append(box)

        for i in range(len(boxes)):
            box = boxes[i]
            utils.drawbox(source.image, box)
            utils.drawlabel(source.image, box)
            
        return InferenceResult(source)
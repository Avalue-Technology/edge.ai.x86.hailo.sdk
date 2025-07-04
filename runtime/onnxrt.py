
import logging
from pathlib import Path
import platform

from typing import List, Sequence

import cv2
import numpy
import numpy.typing
import psutil
import onnxruntime

from arguments import Arguments
from sdk.data.inference_source import InferenceSource


from ..commons import utils

from ..data.bounding_box import BoundingBox
from ..data.coco_80 import find_class_id
from ..data.inference_result import InferenceResult
from ..data.model_information import ModelInformation

from .runtime import Runtime

logger = logging.getLogger(__name__)

class Onnxrt(Runtime):
    
    
    def __init__(
        self,
        args: Arguments,
    ):
        super().__init__(args)

        self._session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(self._model_path)
        
        self._session_inputs = self._session.get_inputs()
        self._input = self._session_inputs[0]
        
        logger.debug(f"input name: {self._input.name}")
        logger.debug(f"input shape: {self._input.shape}")
        logger.debug(f"input type: {self._input.type}")
        
        self._session_outputs = self._session.get_outputs()
        self._output = self._session_outputs[0]
        
        logger.debug(f"output name: {self._output.name}")
        logger.debug(f"output shape: {self._output.shape}")
        logger.debug(f"output type: {self._output.type}")
        
        self._batch_size, self._channels, self._width, self._height = (
            self._input.shape
        )
        
        self._information = ModelInformation(
            Path(self._model_path).name,
            f"CPU {platform.processor()}",
			self._width,
			self._height,
		)
        
    @property
    def temperature(self) -> int:
        coretemp = psutil.sensors_temperatures().get("coretemp")
        
        if (coretemp is not None):
            return int(coretemp[0].current)
        
        return 0
    
    def nmsboxes(
        self,
        bounding_boxes: List[BoundingBox],
        confidence: int,
        threshold: int,
    )  -> Sequence[int]:
        
        bboxes = []
        scores = []
        
        for box in bounding_boxes:
            x1, y1 = box.lefttop()
            x2, y2 = box.rightbottom()
            
            width = x2 - x1
            height = y2 - y1
            bboxes.append([x1, y1, width, height])
            scores.append(box.confidence)
            
        return cv2.dnn.NMSBoxes(
            bboxes,
            scores,
            confidence / 100,
            threshold / 100
        )
    
    def decode84(
        self,
        source: cv2.typing.MatLike,
        outputs: List,
    ) -> BoundingBox:
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
        
        x, y, width, height = outputs[0:4]
        scores = outputs[4:]
        
        box = BoundingBox()
        
        box.id = int(numpy.argmax(scores))
        box.name = find_class_id(None, box.id)
        box.confidence = numpy.amax(scores)
        
        box.center = (
            (x - offset_width) / ratio,
            (y - offset_height) / ratio
        )
        box.width = width / ratio
        box.height = height / ratio
        
        return box
    
    def decode6(
        self,
        source: cv2.typing.MatLike,
        outputs: List,
    ) -> BoundingBox:
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
        box.id = int(id)
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
        
        # image = cv2.resize(
        #     image,
        #     (self._width, self._height),
        #     interpolation=cv2.INTER_CUBIC
        # )
        
        data = numpy.array(image) / 255.0
        data = numpy.transpose(data, (2, 0, 1))
        data = numpy.expand_dims(data, axis=0).astype(numpy.float32)
        
        return data
        
    def inference_image_84(self, source: InferenceSource) -> InferenceResult:
        input_data = self.preprocess(source.image)
        output_data = self._session.run(None, {self._input.name: input_data})
        
        if (not self._display):
            return InferenceResult(source)
        
        outputs = numpy.transpose(numpy.squeeze(output_data[0])) # type: ignore
        rows = outputs.shape[0]
        boxes = []

        for i in range(rows):
            
            classes_scores = outputs[i][4:]
            max_score = numpy.amax(classes_scores)
            
            if max_score >= (source.confidence / 100):
                box = self.decode84(source.image, outputs[i])
                boxes.append(box)
            
        indices = self.nmsboxes(
            boxes,
            source.confidence,
            source.threshold
        )
        
        for i in indices:
            box = boxes[i]
            utils.drawbox(source.image, box)
            utils.drawlabel(source.image, box)
            
        return InferenceResult(source)
    
    def inference_image_6(self, source: InferenceSource) -> InferenceResult:
        input_data = self.preprocess(source.image)
        output_data = self._session.run(None, {self._input.name: input_data})
        
        if (not self._display):
            return InferenceResult(source)

        outputs = numpy.squeeze(output_data[0]) # type: ignore
        rows = len(outputs)
        boxes = []
        
        for i in range(rows):
            score = outputs[i, -2]
            if (score >= (source.confidence / 100.0)):
                box = self.decode6(source.image, outputs[i])
                boxes.append(box)

        indices = self.nmsboxes(
            boxes,
            source.confidence,
            source.threshold
        )
        
        for i in indices:
            box = boxes[i]
            utils.drawbox(source.image, box)
            utils.drawlabel(source.image, box)
        
        return InferenceResult(source)
    
    def inference(self, source: InferenceSource) -> InferenceResult:
        if (self._no_inference):
            return InferenceResult(source)
        
        if self._output.shape[1] == 84:  # YOLOv8-like
            return self.inference_image_84(source)
        
        elif self._output.shape[2] == 6: # YOLOv10-like
            return self.inference_image_6(source)
        
        raise NotImplementedError()
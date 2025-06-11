
from functools import partial

import logging
import threading
import time


from pathlib import Path
from typing import List, Sequence

import numpy
import numpy.typing

import cv2

from arguments import Arguments

from ..commons import utils

from ..runtime.runtimeasync import RuntimeAsync

from ..data.bounding_box import BoundingBox
from ..data.coco_80 import find_class_id
from ..data.inference_source import InferenceSource
from ..data.inference_result import InferenceResult
from ..data.model_information import ModelInformation


from hailo_platform import Device
from hailo_platform.pyhailort.pyhailort import (HEF, ConfiguredInferModel,
                                                InferModel,
                                                Device, VDevice,
                                                HailoSchedulingAlgorithm)

logger = logging.getLogger(__name__)

class HailortAsync(RuntimeAsync):
    
    def __init__(
        self,
        args: Arguments,
    ):
        super().__init__(args)
        
        self._session = HEF(self._model_path)
        logger.debug(f"init {self._session}")
        
       
        # Define dataset params
        self._input_vstream_info = self._session.get_input_vstream_infos()[0]
        self._output_vstream_info = self._session.get_output_vstream_infos()[0]
        self._height, self._width, self._channels = self._input_vstream_info.shape
        logger.debug(f"create stream information {self._height} {self._width} {self._channels}")

        self._device = Device()
        self._device_information = self._device.control.identify()
        self._information = ModelInformation(
            Path(self._model_path).name,
            str(self._device_information.device_architecture),
            self._width,
            self._height,
        )
        logger.debug(f"create device information {self._information}")
        
        self._running = threading.Event()
    
        
    @property
    def information(self) -> ModelInformation:
        return self._information
    
    @property
    def temperature(self) -> int:
        if (not self._device):
            return 0
        
        return int(self._device.control.get_chip_temperature().ts0_temperature)
    
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
        
        x1 *= self._width
        y1 *= self._height
        x2 *= self._width
        y2 *= self._height
        
        x1 -= offset_width
        y1 -= offset_height
        x2 -= offset_width
        y2 -= offset_height
        
        x1 /= ratio
        y1 /= ratio
        x2 /= ratio
        y2 /= ratio
        
        
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
        
        data = numpy.expand_dims(image, axis=0)

        return data.astype(numpy.uint8)

    def postprocess(self, bindings: List[ConfiguredInferModel.Bindings], source: InferenceSource) -> InferenceResult:
        binding = bindings[0]
        output = binding.output()
        outputs = output.get_buffer()
        
        row = len(outputs)
        boxes = []
        
        for i in range(row):
             
            if len(outputs[i]) <= 0:
                continue
            
            for output in outputs[i]:
                
                y1, x1, y2, x2, score = output #! important

                if score >= (source.confidence / 100):
                    box = self.decode6(
                        source.image,
                        [x1, y1, x2, y2, score, i]
                    )
                    boxes.append(box)
        
        indices = self.nmsboxes(
            boxes,
            source.confidence,
            source.threshold
        )
        
    
        for i in indices:
            utils.drawbox(source.image, boxes[i])
            utils.drawlabel(source.image, boxes[i])
            
        return InferenceResult(source)
        
        
    def create_bindings(
        self,
        infermodel: InferModel,
        configured_infermodel: ConfiguredInferModel,
        frame: numpy.typing.NDArray
    ) -> ConfiguredInferModel.Bindings:
        
        bindings: ConfiguredInferModel.Bindings = configured_infermodel.create_bindings()
        bindings.input().set_buffer(frame)
        bindings.output().set_buffer(
            numpy.empty(
                shape=(infermodel.output().shape),
                dtype=numpy.float32,
            )
        )
        return bindings
        
    def callback(self, completion_info, bindings, source: InferenceSource) -> None:
        if completion_info.exception:
            logger.error(f"inference error: {completion_info.exception}")
            
        try:
            if (self._display):
                result: InferenceResult = self.postprocess(bindings, source)
                
            else:
                result: InferenceResult = InferenceResult(source)
                
            self._q_result.put(result)
                
        except Exception as e:
            logger.error(e)
        
    def stop(self) -> None:
        logger.debug(f"stop {self}")
        self._running.clear()
        self.clear()
        
    def exit(self) -> None:
        time.sleep(0.1)
        logger.debug(f"exit {self._device}")
        self._device.release() # type: ignore
        self._device = None
                
    def run(self):
        self._running.set()
                
        logger.debug(f"start grab image")
        
        self._vdevice_params = VDevice.create_params()
        self._vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        
        
        with VDevice(self._vdevice_params) as vdevice:
            logger.debug(f"create vdevice {vdevice}")
            infermodel = vdevice.create_infer_model(
                self._model_path,
                self._session.get_network_group_names()[0]
            )
        
            with infermodel.configure() as configured_infermodel:
            
                while self._running.is_set():
                    
                    source: InferenceSource = self._q_frame.get()
                    if source is None or source.image is None:
                        time.sleep(0.001)
                        continue
                    
                    source.timestamp = time.time()
                    
                    if (self._no_inference):
                        self._q_result.put(InferenceResult(source))
                        time.sleep(0.001)
                        continue
                    
                    bindings = []
                    bindings.append(
                        self.create_bindings(
                            infermodel,
                            configured_infermodel,
                            self.preprocess(source.image)
                        )
                    )
                    
                    try:
                        configured_infermodel.wait_for_async_ready(
                            10 * 1000,
                            1,
                        )
                        
                        job = configured_infermodel.run_async(
                            bindings=bindings,
                            callback=partial(
                                self.callback,
                                bindings=bindings,
                                source=source,
                            ),
                        )
                        job.wait(1000)
                        
                    except Exception as e:
                        logger.error(e)
                        pass
        
        self._running.clear()
        self.clear()
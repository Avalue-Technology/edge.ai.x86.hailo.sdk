
from functools import partial

import logging
import threading
import time

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from typing import List, Optional, Sequence

import numpy
import numpy.typing

import cv2

from ..commons import utils
from ..commons.monitor import Monitor

from ..runtime.runtimeasync import RuntimeAsync

from ..data.circular_buffer import CircularBuffer
from ..data.bounding_box import BoundingBox
from ..data.coco_80 import find_class_id
from ..data.inference_source import InferenceSource
from ..data.inference_result import InferenceResult
from ..data.model_information import ModelInformation


from hailo_platform import Device
from hailo_platform.pyhailort.pyhailort import (HEF, ConfigureParams,
                                                FormatType, FormatOrder,
                                                MipiDataTypeRx, MipiPixelsPerClock,
                                                MipiClockSelection, MipiIspImageInOrder,
                                                MipiIspImageOutDataType, IspLightFrequency, HailoPowerMode,
                                                Endianness, HailoStreamInterface,
                                                InputVStreamParams, OutputVStreamParams,
                                                InputVStreams, OutputVStreams,
                                                InferVStreams, HailoStreamDirection, HailoFormatFlags, HailoCpuId, Device, VDevice,
                                                DvmTypes, PowerMeasurementTypes, SamplingPeriod, AveragingFactor, MeasurementBufferIndex,
                                                HailoRTException, HailoSchedulingAlgorithm, HailoRTStreamAbortedByUser, AsyncInferJob,
                                                HailoCommunicationClosedException)

logger = logging.getLogger(__name__)

class HailortAsync(RuntimeAsync):
    
    def __init__(
        self,
        hef_path: str,
    ):
        super().__init__()
        
        self._hef_path = hef_path
        
        self._session = HEF(hef_path)
        
        self._vdevice_params = VDevice.create_params()
        
        # hailo/hailort/hailort/libhailort/include/hailo/hailort.h:hailo_scheduling_algorithm_e
        self._vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        
        # hailo/hailort/hailort/libhailort/include/hailo/hailort.h:hailo_vdevice_params_t
        # self._vdevice_params.multi_process_service = True
        
        self._vdevice = VDevice(self._vdevice_params)
        
        self._infer_model = self._vdevice.create_infer_model(hef_path, self._session.get_network_group_names()[0])
        self._configured_infer_model = self._infer_model.configure()
        
        self._device = Device()
        self._device_information = self._device.control.identify()
        
        # Define dataset params
        self._input_vstream_info = self._session.get_input_vstream_infos()[0]
        self._output_vstream_info = self._session.get_output_vstream_infos()[0]
        
        self._input_name = self._input_vstream_info.name
        self._input_type = self._input_vstream_info.format.type
        
        logger.debug(self._input_name)
        logger.debug(self._input_type)
        
        self._output_name = self._output_vstream_info.name
        self._output_type = self._output_vstream_info.format.type
        
        logger.debug(self._output_name)
        logger.debug(self._output_type)
                
        logger.debug(self._input_vstream_info)
        logger.debug(self._input_vstream_info.name)
        logger.debug(self._input_vstream_info.format)
        logger.debug(self._input_vstream_info.format.type)
        
        logger.debug(self._output_vstream_info)
        logger.debug(self._output_vstream_info.name)
        logger.debug(self._output_vstream_info.format)
        logger.debug(self._output_vstream_info.format.type)
        
        logger.debug(self._configured_infer_model.get_async_queue_size())
        
        self._capacity: int = self._configured_infer_model.get_async_queue_size()
        
        # self._input_param = InputVStreamParams.make(
        #     self._infer,
        #     format_type=FormatType.UINT8,
        # )
        
        # self._output_param = OutputVStreamParams.make(
        #     self._infer,
        #     format_type=FormatType.FLOAT32
        # )
        
        self._height, self._width, self._channels = self._input_vstream_info.shape
        
        # logger.debug(self._height)
        # logger.debug(self._width)
        
        self._information = ModelInformation(
            Path(hef_path).name,
            str(self._device_information.device_architecture),
            self._width,
            self._height,
        )
        
        self._running = threading.Event()
    
        
    @property
    def information(self) -> ModelInformation:
        return self._information
    
    @property
    def temperature(self) -> int:
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

    def postprocess(self, bindings, source: InferenceSource) -> InferenceSource:
        image = source.image
        binding = bindings[0]
        output = binding.output(self._output_name)
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
                        image,
                        [x1, y1, x2, y2, score, i]
                    )
                    boxes.append(box)
        
        indices = self.nmsboxes(
            boxes,
            source.confidence,
            source.threshold
        )
        
        if (self.display):
            for i in indices:
                box = boxes[i]
                utils.drawbox(image, box)
                utils.drawlabel(image, box)
        
        source.image = image
        return source
        
        
    def create_bindings(self, frame: numpy.typing.NDArray):
        return self._configured_infer_model.create_bindings(
            input_buffers={
                self._input_name: frame,
            },
            output_buffers={
                self._output_name: numpy.empty(
                    shape=self._infer_model.output(self._output_name).shape,
                    dtype=numpy.float32,
                )
            }
        )
        
    def callback(self, completion_info, bindings, source: InferenceSource) -> None:
        if completion_info.exception:
            logger.error(f"inference error: {completion_info.exception}")
            
        try:
            source = self.postprocess(bindings, source)
            self._q_result.put(InferenceResult(source))
                
        except Exception as e:
            # logger.error(e)
            pass
            
    def stop(self) -> None:
        self._running.clear()
        self.clear()
        
        self._configured_infer_model.wait_for_async_ready(
            timeout_ms=10000,
            frames_count=self._capacity
        )
        
    def run(self):
        self._running.set()
                
        logger.debug(f"start grab image")
        
        while self._running.is_set():
            
            source: InferenceSource = self._q_frame.get()
            if source is None or source.image is None:
                time.sleep(0.001)
                continue
            
            bindings = []
            bindings.append(
                self.create_bindings(
                    self.preprocess(source.image)
                )
            )
                
            source.timestamp = time.time()

            try:
                self._configured_infer_model.wait_for_async_ready(
                    10 * 1000,
                    1,
                )
                
                self._configured_infer_model.run_async(
                    bindings=bindings,
                    callback=partial(
                        self.callback,
                        bindings=bindings,
                        source=source,
                    ),
                )
                
            except Exception as e:
                logger.error(e)
                pass
        
        self._running.clear()
        self.clear()
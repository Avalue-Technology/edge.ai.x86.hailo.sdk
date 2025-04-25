
import logging
import time

from typing import Dict, List, Sequence

import numpy
import numpy.typing

import cv2
import psutil

import commons
from data.bounding_box import BoundingBox
from data.coco_80 import find_class_id
from data.inference_result import InferenceResult


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

from data.model_information import ModelInformation
from model.runntime import Runtime

logger = logging.getLogger(__name__)

class Hailort(Runtime):
    
    def __init__(
        self,
        hef_path: str,
        drawbbox: bool,
    ):
        
        self._hef_path = hef_path
        self._drawbbox = drawbbox
        
        self._session = HEF(hef_path)
        
        self._vdevice = VDevice()
        
        self._configure_params = ConfigureParams.create_from_hef(
            self._session,
            interface=HailoStreamInterface.PCIe
        )
        
        self._network_group = self._vdevice.configure(
            self._session,
            self._configure_params
        )[0]
        
        self._network_group_params = self._network_group.create_params()

        # Create input and output virtual streams params
        self._input_vstreams_params = InputVStreamParams.make_from_network_group(
            self._network_group,
            quantized=False,
            format_type=FormatType.UINT8
        )
        
        self._output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self._network_group,
            quantized=False,
            format_type=FormatType.FLOAT32
        )

        # Define dataset params
        self._input_vstream_info = self._session.get_input_vstream_infos()[0]
        self._output_vstream_info = self._session.get_output_vstream_infos()[0]
        
        self._input_name = self._input_vstream_info.name
        
        logger.debug(self._input_vstream_info)
        logger.debug(self._input_vstream_info.name)
        logger.debug(self._input_vstream_info.format)
        logger.debug(self._input_vstream_info.format.type)
        
        logger.debug(self._output_vstream_info)
        logger.debug(self._output_vstream_info.name)
        logger.debug(self._output_vstream_info.format)
        logger.debug(self._output_vstream_info.format.type)
        
        self._height, self._width, self._channels = self._input_vstream_info.shape
        
        logger.debug(self._height)
        logger.debug(self._width)
        
        self._information = ModelInformation(
            self._width,
            self._height,
        )
        
    @property
    def information(self) -> ModelInformation:
        return self._information
    
    
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
        
        ratio = commons.get_ratio(
            shape_width,
            shape_height,
            self._width,
            self._height,
        )
        
        offset_width, offset_height = commons.get_offset_length(
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
    
    
    def preprocess_image(
        self,
        source: cv2.typing.MatLike,
    ) -> numpy.typing.NDArray:
        
        image = commons.preprocess_image(
            source,
            self._width,
            self._height
        )
        
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # data = numpy.array(image)
        # data = numpy.transpose(data, (2, 0, 1))
        data = numpy.expand_dims(image, axis=0)

        return data.astype(numpy.uint8)
        

    def inference(self, source: cv2.typing.MatLike, confidence: int, threshold: int) -> InferenceResult:

        image = source
        
        input_data = self.preprocess_image(source)
        
        now = time.time()
        output_data = None
        with InferVStreams(
            self._network_group,
            self._input_vstreams_params,
            self._output_vstreams_params
        ) as infer_pipeline:
            
            with self._network_group.activate(self._network_group_params):
                output_data = infer_pipeline.infer({self._input_name: input_data})
                cpu_usage = psutil.cpu_percent()
                
        end = time.time()
        spendtime = end - now
        
        if (not self._drawbbox):
            return InferenceResult(
                spendtime,
                cpu_usage,
                image,
            )
        
        key = list(output_data.keys())[0]
        outputs = output_data[key][0]
        row = len(outputs)
        
        boxes = []
        
        for i in range(row):
            
            if len(outputs[i]) <= 0:
                continue
            
            for output in outputs[i]:
                
                y1, x1, y2, x2, score = output #! important

                if score >= (confidence / 100):
                    box = self.decode6(
                        source, 
                        [x1, y1, x2, y2, score, i]
                    )
                    boxes.append(box)
        
        indices = self.nmsboxes(
            boxes,
            confidence,
            threshold
        )
        
        for i in indices:
            box = boxes[i]
            commons.drawbox(image, box)
            commons.drawlabel(image, box)
            
        return InferenceResult(
            spendtime,
            cpu_usage,
            image,
        )

import numpy
import logging
import time

from typing import Dict
from collections import defaultdict

import cv2
import psutil

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

logger = logging.getLogger(__name__)

class Hailort:
    
    def __init__(
        self,
        hef_path: str
    ):
        
        self._session = HEF(hef_path)
        
        self._vdevice = VDevice()
        
        self._configure_params = ConfigureParams.create_from_hef(self._session, interface=HailoStreamInterface.PCIe)
        self._network_group = self._vdevice.configure(self._session, self._configure_params)[0]
        self._network_group_params = self._network_group.create_params()
            

        # Create input and output virtual streams params
        self._input_vstreams_params = InputVStreamParams.make(self._network_group, format_type=FormatType.FLOAT32)
        self._output_vstreams_params = OutputVStreamParams.make(self._network_group, format_type=FormatType.FLOAT32)

        # Define dataset params
        self._input_vstream_info = self._session.get_input_vstream_infos()[0]
        self._output_vstream_info = self._session.get_output_vstream_infos()[0]
        
        self._input_name = self._input_vstream_info.name
        
        logger.debug(self._input_vstream_info)
        logger.debug(self._output_vstream_info)
        
        self._height, self._width, self._channels = self._input_vstream_info.shape
        
    def __nms_numpy__(self, boxes, scores, threshold):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = numpy.maximum(x1[i], x1[order[1:]])
            yy1 = numpy.maximum(y1[i], y1[order[1:]])
            xx2 = numpy.minimum(x2[i], x2[order[1:]])
            yy2 = numpy.minimum(y2[i], y2[order[1:]])

            w = numpy.maximum(0.0, xx2 - xx1 + 1)
            h = numpy.maximum(0.0, yy2 - yy1 + 1)
            
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)

            inds = numpy.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep
    
    def __nms_format__(self, boxes, classids, confidences, confidence):
        
        result_boxes = []
        result_scores = []
        result_classids = []
        
        for i in range(len(boxes)):
            if(confidences[i] < confidence):
                continue
            
            x_c, y_c, w, h = boxes[i]
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2
            
            result_boxes.append([x1, y1, x2, y2])
            result_scores.append(confidences[i])
            result_classids.append(classids[i])
            
        return (
            numpy.array(result_boxes),
            numpy.array(result_scores),
            numpy.array(result_classids),
        )

    def inference_image(self, source: cv2.typing.MatLike, confidence: int, threshold: int) -> InferenceResult:
        image: cv2.typing.MatLike = cv2.resize(source, (self._width, self._height))
        image = image.transpose(2, 0, 1)
        
        input_data = image.astype(numpy.float32)
        input_data = numpy.expand_dims(input_data, axis=0)
        
        now = time.time()
        outputs = None
        with InferVStreams(self._network_group, self._input_vstreams_params, self._output_vstreams_params) as infer_pipeline:
            with self._network_group.activate(self._network_group_params):
                outputs = infer_pipeline.infer({self._input_name: input_data})
                cpu_usage = psutil.cpu_percent()
        end = time.time()
        
        # 取得唯一一個 output 的名稱（例如 'yolov8l/yolov8_nms_postprocess'）
        output_key = list(outputs.keys())[0]
        class_boxes = outputs[output_key]  # 是個 list，長度為 class 數
        
        logger.debug(output_key)
        logger.debug(class_boxes)
        

        results = []
        for class_id, boxes in enumerate(class_boxes):
            
            logger.debug(f"class_id: {class_id}")
            logger.debug(f"boxes: {boxes}")
        
        logger.debug(results)
        return None
        
        output = outputs[0]  # [1, 84, 8400]
        output = numpy.squeeze(output)  # [84, 8400]
        output = output.transpose(1, 0)  # [8400, 84]

        output_boxes = output[:, :4]  # cx, cy, w, h
        output_scores = output[:, 4:]  # class confidences
        output_classids = numpy.argmax(output_scores, axis=1)
        output_confidences = numpy.max(output_scores, axis=1)
        
        boxes, scores, classids = (
            self.__nms_format__(
                output_boxes,
                output_classids,
                output_confidences,
                confidence / 100
            )
        )
        
        final_boxes = []
        final_scores = []
        final_classids = []
        
        for clsid in numpy.unique(classids):
            idxs = numpy.where(classids == clsid)[0]
            cls_boxes = boxes[idxs]
            cls_scores = scores[idxs]

            keep = self.__nms_numpy__(
                cls_boxes,
                cls_scores,
                threshold / 100
            )
            final_boxes.extend(cls_boxes[keep])
            final_scores.extend(cls_scores[keep])
            final_classids.extend([clsid] * len(keep))
        

        for i in range(len(final_boxes)):
            box = final_boxes[i]
            score = final_scores[i]
            id = final_classids[i]
            
            # name = find_class_id(None, id)
            name = find_class_id(("vehicle",), id)
            # name = find_class_id(("road", "vehicle"), id)
            # name = find_class_id(("food", "daily", "electronics"), id)
            # name = find_class_id(("food",), id)
            
            if name is None:
                continue
            
            x1 = int(box[0] * source.shape[1] / self._width)
            y1 = int(box[1] * source.shape[0] / self._height)
            x2 = int(box[2] * source.shape[1] / self._width)
            y2 = int(box[3] * source.shape[0] / self._height)

            cv2.rectangle(
                source,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                source,
                f"{id} {name} {score:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            
        spendtime = end - now
        
        return InferenceResult(
            spendtime,
            cpu_usage,
            source,
        )
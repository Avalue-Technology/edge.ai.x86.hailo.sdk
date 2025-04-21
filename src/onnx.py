
import numpy
import logging
import time

from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
import psutil
import onnxruntime

from data.coco_80 import find_class_id
from data.inference_result import InferenceResult

logger = logging.getLogger(__name__)

class Onnxrt():
    
    
    def __init__(
        self,
        onnx_path: str,
    ):

        self._onnx_path = onnx_path
        self._session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(self._onnx_path)
        
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
        
    def draw_detections(self, img: numpy.ndarray, box: List[float], score: float, class_id: int, class_name: str) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (List[float]): Detected bounding box coordinates [x, y, width, height].
            score (float): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)

        # Create the label text with class name and score
        label = f"{class_id} {class_name} {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, 
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            (0, 255, 0),
            cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    
    def output_process(self, gain, boxes, classids, confidences, confidence):
        
        result_boxes = []
        result_scores = []
        result_classids = []
        
        
        for i in range(len(boxes)):
            if(confidences[i] < confidence):
                continue
            
            logger.debug(f"confidences[i]{confidences[i]} >= confidence{confidence}")
            
            # x_c, y_c, w, h = boxes[i]
            # x1 = x_c - w / 2
            # y1 = y_c - h / 2
            # x2 = x_c + w / 2
            # y2 = y_c + h / 2
            # result_boxes.append([x1, y1, x2, y2])
            
            x, y, w, h = boxes[i]
            left = int((x - w / 2) / gain)
            top = int((y - h / 2) / gain)
            width = int(w / gain)
            height = int(h / gain)
            result_boxes.append([left, top, width, height])
            
            result_scores.append(confidences[i])
            result_classids.append(classids[i])
            
        return (
            numpy.array(result_boxes),
            numpy.array(result_scores),
            numpy.array(result_classids),
        )
        
    def letterbox(
        self,
        img: numpy.ndarray,
        new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[numpy.ndarray, float, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, r, (left, top)
        
    def inference_image_v8(
        self,
        source: cv2.typing.MatLike,
        confidence: int,
        threshold: int,
    ) -> InferenceResult:
        source_height, source_width = source.shape[:2]
        image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        
        # image = cv2.resize(image, (self._width, self._height))
        image, gain, pad = self.letterbox(image, (self._width, self._height))

        input_data = numpy.array(image) / 255.0
        input_data = numpy.transpose(image, (2, 0, 1))
        input_data = numpy.expand_dims(input_data, axis=0).astype(numpy.float32)
        
        now = time.time()
        outputs = self._session.run(None, {self._input.name: input_data})
        cpu_usage = psutil.cpu_percent()
        end = time.time()
        spendtime = end - now
        
        # Transpose and squeeze the output to match the expected shape
        output = numpy.transpose(numpy.squeeze(outputs[0]))
        
        # Get the number of rows in the output array
        rows = output.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        output[:, 0] -= pad[0]
        output[:, 1] -= pad[1]

        # Iterate over each row in the output array
        for i in range(rows):
            
            # logger.debug(f"output[{i}]: {output[i]}")
            
            # Extract the class scores from the current row
            classes_scores = output[i][4:]

            # Find the maximum score among the class scores
            max_score = numpy.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= (confidence / 100):
                # Get the class ID with the highest score
                class_id = numpy.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = output[i][0], output[i][1], output[i][2], output[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            confidence / 100,
            threshold / 100
        )
        
        for i in indices:
            box = boxes[i]
            score = scores[i]
            id = class_ids[i]
            
            name = find_class_id(None, id)
            # name = find_class_id(("vehicle",), id)
            # name = find_class_id(("road", "vehicle"), id)
            # name = find_class_id(("food", "daily", "electronics"), id)
            # name = find_class_id(("food",), id)
            
            self.draw_detections(source, box, score, id, name)
            
        return InferenceResult(
            spendtime,
            cpu_usage,
            source,
        )
        
    
    def inference_image_v10(
        self,
        source: cv2.typing.MatLike,
        confidence: int,
        threshold: int,
    ) -> InferenceResult:
        image = source
        
        source_height, source_width = source.shape[:2]
        # image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        
        # image = cv2.resize(image, (self._width, self._height))
        image, gain, pad = self.letterbox(image, (self._width, self._height))
        pad_x, pad_y = pad
        
        # logger.debug(f"pad: {pad}")
        # logger.debug(f"gain: {gain}")
        
        input_data = numpy.array(image) / 255.0
        input_data = numpy.transpose(image, (2, 0, 1))
        input_data = numpy.expand_dims(input_data, axis=0).astype(numpy.float32)
        
        now = time.time()
        outputs = self._session.run(None, {self._input.name: input_data})
        cpu_usage = psutil.cpu_percent()
        end = time.time()
        spendtime = end - now
        
        # Transpose and squeeze the output to match the expected shape
        output = numpy.squeeze(outputs[0])
        boxes = output[:, :-2]
        scores = output[:, -2]
        class_ids = output[:, -1].astype(int)
        
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            confidence / 100,
            threshold / 100
        )
        
        # logger.debug(f"output: {len(output)}")
        # logger.debug(f"boxes: {boxes}")
        # logger.debug(f"scores: {scores}")
        # logger.debug(f"class_ids: {class_ids}")
        
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            conf = scores[i]
            clsid = class_ids[i]
            
            # logger.debug(f"boxes[{i}]: {boxes[i]}")
            # logger.debug(f"conf[{i}]: {scores[i]}")
            # logger.debug(f"clsid[{i}]: {class_ids[i]}")
            
            if (conf < confidence / 100.0):
                continue
            
            name = find_class_id(None, clsid)
            # name = find_class_id(("vehicle",), id)
            # name = find_class_id(("road", "vehicle"), id)
            # name = find_class_id(("food", "daily", "electronics"), id)
            # name = find_class_id(("food",), id)

            # Create the label text with class name and score
            label = f"{clsid} {name} {conf:.2f}"
            
            # logger.debug(f"{x1}, {y1}, {x2}, {y2}")
            
            x1 = (x1 - pad_x) / gain
            y1 = (y1 - pad_y) / gain
            x2 = (x2 - pad_x) / gain
            y2 = (y2 - pad_y) / gain
            
            # logger.debug(f"{x1}, {y1}, {x2}, {y2}")
            
            # x1 /= self._width
            # y1 /= self._height
            # x2 /= self._width
            # y2 /= self._height
            
            # x1 *= source_width
            # y1 *= source_height
            # x2 *= source_width
            # y2 *= source_height

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
            
            cv2.rectangle(
                source,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            
            cv2.rectangle(
                source, 
                (int(label_x), int(label_y - label_height)),
                (int(label_x + label_width), int(label_y + label_height)),
                (0, 255, 0),
                cv2.FILLED
            )

            # Draw the label text on the image
            cv2.putText(
                source,
                label,
                (int(label_x), int(label_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        
        return InferenceResult(
            spendtime,
            cpu_usage,
            source,
        )
            
    
    
    def inference_image(
        self,
        source: cv2.typing.MatLike,
        confidence: int,
        threshold: int,
    ) -> InferenceResult:
        
        if self._output.shape[1] == 84:  # YOLOv8-like
            return self.inference_image_v8(source, confidence, threshold)
        
        elif self._output.shape[2] == 6: # YOLOv10-like
            return self.inference_image_v10(source, confidence, threshold)
        

from abc import ABC, abstractmethod

import cv2

from data.inference_result import InferenceResult
from data.model_information import ModelInformation


class Runtime():
    
    def __init__(self) -> None:
        pass
    
    
    @property
    @abstractmethod
    def information(self) -> ModelInformation:
        pass
    
    @abstractmethod
    def inference(
        self,
        image: cv2.typing.MatLike,
        confidence: int,
        threshold: int,
    ) -> InferenceResult:
        pass
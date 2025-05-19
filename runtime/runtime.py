
from abc import ABC, abstractmethod

import cv2

from sdk.data.inference_source import InferenceSource

from ..data.inference_result import InferenceResult
from ..data.model_information import ModelInformation


class Runtime():
    
    def __init__(self) -> None:
        self._information: ModelInformation = ModelInformation("undefined", "undefined", 0, 0)
        self._temperature: int = 0
        self._display: bool = False
        
        self._spendtime = 0.0
        self._fps = 0.0
    
    @property
    def information(self) -> ModelInformation:
        return self._information
    
    @information.setter
    def information(self, information: ModelInformation) -> None:
        self._information = information
    
    @property
    def temperature(self) -> int:
        return self._temperature
    
    @property
    def display(self) -> bool:
        return self._display
    
    @display.setter
    def display(self, display: bool) -> None:
        self._display = display
    
    @property
    def spendtime(self) -> float:
        return self._spendtime
    
    @spendtime.setter
    def spendtime(self, spendtime: float) -> None:
        self._spendtime = spendtime
        
    @property
    def fps(self) -> float:
        return self._fps
    
    @fps.setter
    def fps(self, fps: float):
        self._fps = fps
    
    @abstractmethod
    def inference(self, source: InferenceSource) -> InferenceResult:
        pass
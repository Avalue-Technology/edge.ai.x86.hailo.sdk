
from abc import ABC, abstractmethod

import logging

import cv2

from sdk.data.circular_buffer import CircularBuffer

from .runtime import Runtime

from ..data.inference_source import InferenceSource
from ..data.inference_result import InferenceResult
from ..data.model_information import ModelInformation

logger = logging.getLogger(__name__)

class RuntimeAsync(Runtime):
    
    def __init__(self) -> None:
        super().__init__()
        
        self._running: bool = False
        
        self._q_frame = CircularBuffer(64)
        self._q_output = CircularBuffer(512)
    
    @property
    def running(self) -> bool:
        return self._running
    
    @running.setter
    def running(self, running: bool) -> None:
        self._running = running
        
    def clear(self) -> None:
        self._q_frame.clear()
        self._q_output.clear()
    
    @abstractmethod
    def put(self, source: InferenceSource) -> None:
        # logger.debug(f"queue frame put: {source.timestamp}")
        self._q_frame.put(source)
    
    @abstractmethod
    def get(self) -> InferenceResult:
        output: InferenceResult = self._q_output.get()
        if (output is None):
            return None
        
        # logger.debug(f"queue output get: {output.timestamp}[{output.spendtime}]")
        return output
    
    @abstractmethod
    def avaliable(self) -> bool:
        return not self._q_frame.isfull
    
    @abstractmethod
    def stop(self) -> None:
        pass
    
    @abstractmethod
    def start(self) -> None:
        pass
    
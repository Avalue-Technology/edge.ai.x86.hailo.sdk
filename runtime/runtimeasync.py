
from abc import ABC, abstractmethod

import logging
import threading

import cv2

from .runtime import Runtime
from ..commons.monitor import Monitor

from ..data.circular_buffer import CircularBuffer
from ..data.inference_source import InferenceSource
from ..data.inference_result import InferenceResult
from ..data.model_information import ModelInformation

logger = logging.getLogger(__name__)

class RuntimeAsync(Runtime):
    
    def __init__(self, size = 64) -> None:
        super().__init__()
        
        self._running: threading.Event = threading.Event()
        self._size = size
        
        self._q_frame = CircularBuffer(size)
        self._q_result = CircularBuffer(size)
    
    @property
    def running(self) -> threading.Event:
        return self._running
        
    @property
    def size(self) -> int:
        return self._size
    
    @size.setter
    def size(self, size: int) -> None:
        self._size = size
        self._q_frame = CircularBuffer(size)
        self._q_result = CircularBuffer(size)
        
    @property
    def avaliable(self) -> int:
        return self._q_frame.avaliable
        
    def clear(self) -> None:
        self._q_frame.clear()
        self._q_result.clear()
    
    @abstractmethod
    def put(self, source: InferenceSource) -> None:
        self._q_frame.put(source)
    
    @abstractmethod
    def get(self) -> InferenceResult:
        return self._q_result.get()
    
    @abstractmethod
    def stop(self) -> None:
        pass
    
    @abstractmethod
    def run(self) -> None:
        pass
    
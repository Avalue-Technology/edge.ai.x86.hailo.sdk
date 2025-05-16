
from enum import Enum
from typing import Any

import logging
import threading
import numpy

logger = logging.getLogger(__name__)

class CircularBufferMethod(Enum):
    
    FIFO = 0
    "First in first out"
    
    LIFO = 1
    "Last in first out"

class CircularBuffer():
    def __init__(
        self, 
        size: int = 5,
        method: CircularBufferMethod = CircularBufferMethod.FIFO
    ):
        self.size = size
        self.method = method
        
        self.buffer = numpy.empty(size, dtype=object)
        self.write_index = 0
        self.read_index = 0
        self.isfull = False
        
        self.lock = threading.Lock()
    
    def __move_read_index__(self) -> None:
        self.read_index = (self.read_index + 1) % self.size

    def __move_write_index__(self) -> None:
        self.write_index = (self.write_index + 1) % self.size
        
    def __add__(self, data: Any) -> int:
        with self.lock:
            # logger.debug(f"write index: {self.write_index}")
            self.buffer[self.write_index] = data
            self.__move_write_index__()

            if self.write_index == self.read_index:
                self.isfull = True
                self.__move_read_index__()

            else:
                self.isfull = False
        
            return self.write_index
        
        
    def __next__(self) -> Any:
        with self.lock:
            if self.write_index == self.read_index and not self.isfull:
                return None

            result = self.buffer[self.read_index]
            # logger.debug(f"read index: {self.read_index}")
            
            self.__move_read_index__()
            self.isfull = False

            return result
        
    def __last__(self) -> Any:
        index = 0
        if self.read_index - 1 == 0:
            index = self.size - 1
            
        else:
            index = self.read_index
        
        # logger.debug(f"read index: {index}")
        result = self.buffer[index]
        return result
    
    def put(self, data: Any) -> int:
        return self.__add__(data)
    
    def next(self) -> Any:
        return self.__next__()
    
    def last(self) -> Any:
        return self.__last__()
    
    def get(self) -> Any:
        if self.method == CircularBufferMethod.FIFO:
            return self.__next__()
        
        if self.method == CircularBufferMethod.LIFO:
            return self.__last__()
            
    def clear(self) -> None:
        self.buffer = numpy.empty(self.size, dtype=object)
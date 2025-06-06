
import heapq
from enum import Enum
from typing import Any, List, Tuple

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
        self._size = size
        self._method = method
        
        self._buffer = numpy.empty(self._size, dtype=object)
        self._write_index = 0
        self._read_index = 0
        self._isfull = False
        
        self._lock = threading.Lock()
    
    def __move_read_index__(self) -> None:
        self._read_index = (self._read_index + 1) % self._size

    def __move_write_index__(self) -> None:
        self._write_index = (self._write_index + 1) % self._size
    
    def __add__(self, data: Any) -> int:
        # logger.debug(f"write[{self._write_index}]: {type(data)}")
        self._buffer[self._write_index] = data
        self.__move_write_index__()

        if self._write_index == self._read_index:
            self._isfull = True
            self.__move_read_index__()

        else:
            self._isfull = False
    
        return self._write_index
        
    def __next__(self) -> Any:
        if self._write_index == self._read_index and not self._isfull:
            return None

        result = self._buffer[self._read_index]
        # logger.debug(f"read[{self._read_index}]: {type(result)}")
        
        self.__move_read_index__()
        self._isfull = False

        return result
    
    def __last__(self) -> Any:
        index = 0
        if self._read_index - 1 == 0:
            index = self._size - 1
            
        else:
            index = self._read_index
        
        result = self._buffer[index]
        # logger.debug(f"read[{index}]: {result} {len(result)}")
        return result
    
    @property
    def avaliable(self) -> int:
        with self._lock:
            return ((self._read_index + self._write_index) % self._size) + 1
    
    def put(self, data: Any) -> int:
        with self._lock:
            return self.__add__(data)
    
    def next(self) -> Any:
        with self._lock:
            return self.__next__()
    
    def last(self) -> Any:
        with self._lock:
            return self.__last__()
    
    def get(self) -> Any:
        with self._lock:
            if self._method == CircularBufferMethod.FIFO:
                return self.__next__()
            
            if self._method == CircularBufferMethod.LIFO:
                return self.__last__()
    
    def clear(self) -> None:
        with self._lock:
            count = 0
            for i in range(self._size):
                if (self._buffer is not None):
                    count += 1
                    
            logger.debug(f"clear {count} items at {repr(self._buffer)[:20]}... ")
            
            for i in range(self._size):
                self._buffer[i] = None
            del self._buffer
            
            self._buffer = numpy.empty(self._size, dtype=object)
            self._write_index = 0
            self._read_index = 0
            self._isfull = False
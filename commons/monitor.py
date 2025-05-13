


import threading
import time
import logging
import numpy

from abc import abstractmethod
from collections import deque
from typing import Deque


from ..data.model_information import ModelInformation

logger = logging.getLogger(__name__)

class Monitor():
    
    def __init__(self, modelname: str):
        
        self._modelname = modelname
        
        self._times = 0

        self._frametotal = 0
        
        self._framecount = 0
        self._framecounts: Deque[int] = deque([0], maxlen=60)
        
        self._spendtimes:Deque[float] = deque([0.0], maxlen=60)
        
        self._task = None
    
    @abstractmethod
    def get_temperature(self) -> int:
        return -1
    
    @abstractmethod
    def get_information(self) -> ModelInformation:
        return ModelInformation("none", "none", 0, 0)

    def task_framecount_reset(self) -> None:
        
        info = self.get_information()
        
        while(True):
            self._times += 1
            
            if (self.framecount > 0 and self.spandtime > 0):
                maxfps = round(1 / self.spandtime, 1)
                latency = int(self.spandtime * 1000)
                logger.info(f"{self._modelname}[{self._frametotal:09d}|{self._times}] fps: {self.framecount:.1f} [{maxfps}/{latency}ms] tempature[{info.device}]: {self.get_temperature()}")
                
            self._framecounts.append(self._framecount)
            self._framecount = 0
            time.sleep(1)
            
    def add_count(self) -> None:
        self._frametotal += 1
        self._framecount += 1
        
    def add_spendtime(self, spendtime: float) -> None:
        self._spendtimes.append(spendtime)
        
    def start(self) -> None:
        if (self._task is None):
            self._task = threading.Thread(target=self.task_framecount_reset, daemon=True)
            
        self._task.start()
        
    def stop(self) -> None:
        if (self._task is not None):
            self._task.join(1)
        
    @property
    def framecount(self):
        return float(numpy.mean(numpy.array(self._framecounts)))
    
    @property
    def spandtime(self) -> float:
        return float(numpy.mean(numpy.array(self._spendtimes)))
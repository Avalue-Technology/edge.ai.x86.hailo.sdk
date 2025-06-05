
import sys
import threading
import time
import logging
import traceback
import numpy
import gc

from pympler import summary, muppy, asizeof

from abc import abstractmethod
from collections import deque
from typing import Deque


from ..data.model_information import ModelInformation

logger = logging.getLogger(__name__)

class Monitor():
    
    def __init__(self, modelname: str = "dryrun"):
        
        self._modelname = modelname
        
        self._times = 0
        self._frametotal = 0
        self._framecount = 0
        self._framecounts: Deque[int] = deque([0], maxlen=60)
        self._spendtimes: Deque[float] = deque([0.0], maxlen=60)
        self._fps = 0.0
        self._latency = 0.0
        
        self._monitor = None
        self._monitor_event = threading.Event()
        
        self._topn = 10
    
    @abstractmethod
    def get_temperature(self) -> int:
        return -1
    
    @abstractmethod
    def get_information(self) -> ModelInformation:
        return ModelInformation("none", "none", 0, 0)
    
    def _dump_summary(self):
        all_objects = muppy.get_objects()
        total_str = sum(1 for o in all_objects if isinstance(o, str))
        total_list = sum(1 for o in all_objects if isinstance(o, list))
        logger.info(f"[MemoryWatchdog] total str: {total_str}, total list: {total_list}")

    def _dump_top_str(self):
        str_objects = [o for o in muppy.get_objects() if isinstance(o, str)]
        sorted_strs = sorted(str_objects, key=asizeof.asizeof, reverse=True)[:self._topn]
        logger.info(f"[MemoryWatchdog] Top {self._topn} largest str:")
        for i, s in enumerate(sorted_strs):
            try:
                logger.debug(f"[{i}] size={asizeof.asizeof(s)/1024:.2f} KB, len={len(s)}, content={repr(s[:100])}")
            except Exception:
                pass

    def _dump_top_list(self):
        list_objects = [o for o in muppy.get_objects() if isinstance(o, list)]
        sorted_lists = sorted(list_objects, key=asizeof.asizeof, reverse=True)[:self._topn]
        logger.info(f"[MemoryWatchdog] Top {self._topn} largest list:")
        for i, lst in enumerate(sorted_lists):
            try:
                logger.debug(f"[{i}] size={asizeof.asizeof(lst)/1024:.2f} KB, len={len(lst)}")
                if len(lst) > 0:
                    logger.debug(f"    id: {id(lst)}, type: {type(lst)}")
                    logger.debug(f"    items: {[repr(x)[:40] for x in lst[:20]]}")
                    
                    refs = gc.get_referrers(lst)
                    for r in refs[:10]:
                        logger.debug(f"        ref by: {type(r)}, preview: {repr(r)[:100]}")
                        try:
                            if isinstance(r, dict) and '__module__' in r:
                                logger.debug(f"            ref by module: {r['__module__']}")
                            elif isinstance(r, (list, tuple, set, dict)):
                                for stack in traceback.extract_stack():
                                    logger.debug(f"            {stack}")
                        except:
                            pass
            except Exception:
                pass

    def __task_monitor__(self) -> None:
        
        info = self.get_information()
        err = 0
        while(self._monitor_event.is_set()):
            self._times += 1
            
            if (
                len(self._framecounts) > 0
                and len(self._spendtimes) > 0
            ):
                self._fps = float(numpy.mean(numpy.array(self._framecounts)))
                self._latency = float(numpy.mean(numpy.array(self._spendtimes)))
                logger.info(f"{self._modelname}[{self._frametotal:09d}|{self._times}|{err}] fps: {self._fps:.1f}({round(self._latency * 1000)}ms) tempature[{info.device}]: {self.get_temperature()}")
                
                
            self._framecounts.append(self._framecount)
            self._framecount = 0
            
            if (self._fps == 0):
                err += 1
                
            else:
                err = 0
                
            if (err > 10):
                break
            
            time.sleep(1)
            

        for thread in threading.enumerate():
            logger.debug(f"Thread name: {thread.name}, ID: {thread.ident}, Alive: {thread.is_alive()}, Daemon: {thread.daemon}")

            
        logger.error(f"montir terminate on error {err}")
        
    def dump_objects(self):
        self._dump_summary()
        self._dump_top_list()
            
    def add_count(self) -> None:
        self._frametotal += 1
        self._framecount += 1
        
    def add_spendtime(self, spendtime: float) -> None:
        self._spendtimes.append(spendtime)
        
    def reset(self) -> None:
        self._times = 0
        self._frametotal = 0
        self._framecount = 0
        self._framecounts.clear()
        self._spendtimes.clear()
        self._fps = 0.0
        self._latency = 0.0
        
    def start(self) -> None:
        if (self._monitor is None):
            self._monitor = threading.Thread(target=self.__task_monitor__, daemon=True)
        
        self._monitor_event.set()
        self._monitor.start()
        
    def stop(self) -> None:
        if (self._monitor is not None):
            self._monitor_event.clear()
            self._monitor.join(1)
        
        self._monitor = None
        
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def latency(self) -> float:
        return self._latency
    
    @property
    def count(self) -> int:
        return self._frametotal

from abc import abstractmethod

from arguments import Arguments

from ..commons.monitor import Monitor

from ..data.inference_source import InferenceSource
from ..data.inference_result import InferenceResult
from ..data.model_information import ModelInformation


class Runtime():
    
    def __init__(self, args: Arguments) -> None:
        self._information: ModelInformation = ModelInformation("undefined", "undefined", 0, 0)
        self._temperature: int = 0
        self._spendtime = 0.0
        self._fps = 0.0
        
        self._display: bool = args.display
        self._model_path = args.model_path
        self._no_inference = args.no_inference
        
    @property
    def monitor(self) -> Monitor:
        return self._monitor
    
    @monitor.setter
    def monitor(self, monitor: Monitor) -> None:
        self._monitor = monitor
        self._monitor.get_temperature = lambda : self.temperature
        self._monitor.get_information = lambda : self.information
                
    def reset(self) -> None:
        self._monitor.reset()
        
    def add_count(self) -> None:
        self._monitor.add_count()
        
    def add_spendtime(self, time: float) -> None:
        self._monitor.add_spendtime(time)
        
    def get_temperature(self) -> int:
        return self.temperature
    
    def get_information(self) -> ModelInformation:
        return self.information
    
    @property
    def latency(self) -> float:
        return self._monitor.latency
    
    @property
    def fps(self) -> float:
        return self._monitor.fps
            
    @property
    def information(self) -> ModelInformation:
        return self._information
    
    @information.setter
    def information(self, information: ModelInformation) -> None:
        self._information = information
    
    @property
    def temperature(self) -> int:
        return self._temperature

    @abstractmethod
    def inference(self, source: InferenceSource) -> InferenceResult:
        pass
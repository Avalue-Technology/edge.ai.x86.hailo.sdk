
from pathlib import Path

from .commons.monitor import Monitor

from .runtime.runtime import Runtime
from .runtime.runtimeasync import RuntimeAsync

from .runtime.onnxrt import Onnxrt
from .runtime.tflitert import Tflitert
from .runtime.hailort import Hailort

from .runtime.hailortasync import HailortAsync


def loadonnx(monitor: Monitor, onnxpath: str) -> Runtime:
    return Onnxrt(monitor, onnxpath)

def loadhef(monitor: Monitor, hefpath: str) -> Runtime:
    return Hailort(monitor, hefpath)

def loadhefasync(monitor: Monitor, hefpath: str) -> RuntimeAsync:
    return HailortAsync(monitor, hefpath)

def loadtflite(monitor: Monitor, tflitepath: str) -> Runtime:
    return Tflitert(monitor, tflitepath)

def loadmodel(monitor: Monitor, model_path: str) -> Runtime:
    mp = Path(model_path)

    if (mp.suffix.lower() == ".onnx"):
        return loadonnx(monitor, model_path)
        
    elif (mp.suffix.lower() == ".hef"):
        return loadhef(monitor, model_path)
    
    elif (mp.suffix.lower() == ".tflite"):
        return loadtflite(monitor, model_path)
    
    raise ValueError(f"unsupport model type: {mp.suffix}")

def loadmodelasync(monitor: Monitor, model_path: str) -> RuntimeAsync:
    mp = Path(model_path)

    if (mp.suffix.lower() == ".hef"):
        return loadhefasync(monitor, model_path)
        
    raise ValueError(f"unsupport model type: {mp.suffix}")
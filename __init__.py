
from pathlib import Path
from typing import Union


from .commons.monitor import Monitor

from .runtime.runtime import Runtime
from .runtime.runtimeasync import RuntimeAsync

from .runtime.onnxrt import Onnxrt
from .runtime.tflitert import Tflitert
from .runtime.hailortasync import HailortAsync


def loadonnx(onnxpath: str) -> Runtime:
    return Onnxrt(onnxpath)

def loadhefasync(hefpath: str) -> RuntimeAsync:
    return HailortAsync(hefpath)

def loadtflite(tflitepath: str) -> Runtime:
    return Tflitert(tflitepath)

def loadmodel(model_path: str) -> Union[Runtime, RuntimeAsync]:
    mp = Path(model_path)

    if (mp.suffix.lower() == ".onnx"):
        return loadonnx(model_path)
    
    elif (mp.suffix.lower() == ".tflite"):
        return loadtflite(model_path)
        
    elif (mp.suffix.lower() == ".hef"):
        return loadhefasync(model_path)
    
    raise ValueError(f"unsupport model type: {mp.suffix}")


from pathlib import Path
from .runtime.runtime import Runtime
from .runtime.runtimeasync import RuntimeAsync

from .runtime.onnxrt import Onnxrt
from .runtime.tflitert import Tflitert
from .runtime.hailort import Hailort

from .runtime.hailortasync import HailortAsync


def loadonnx(onnxpath: str) -> Runtime:
    return Onnxrt(onnxpath)

def loadhef(hefpath: str) -> Runtime:
    return Hailort(hefpath)

def loadhefasync(hefpath: str) -> RuntimeAsync:
    return HailortAsync(hefpath)

def loadtflite(tflitepath: str) -> Runtime:
    return Tflitert(tflitepath)

def loadmodel(model_path: str) -> Runtime:
    mp = Path(model_path)

    if (mp.suffix.lower() == ".onnx"):
        return loadonnx(model_path)
        
    elif (mp.suffix.lower() == ".hef"):
        return loadhef(model_path)
    
    elif (mp.suffix.lower() == ".tflite"):
        return loadtflite(model_path)
    
    raise ValueError(f"unsupport model type: {mp.suffix}")

def loadmodelasync(model_path: str) -> RuntimeAsync:
    mp = Path(model_path)

    if (mp.suffix.lower() == ".hef"):
        return loadhefasync(model_path)
        
    raise ValueError(f"unsupport model type: {mp.suffix}")
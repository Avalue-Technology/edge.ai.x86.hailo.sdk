
from pathlib import Path
from typing import Union

from arguments import Arguments


from .commons.monitor import Monitor

from .runtime.runtime import Runtime
from .runtime.runtimeasync import RuntimeAsync

from .runtime.onnxrt import Onnxrt
from .runtime.tflitert import Tflitert
from .runtime.hailortasync import HailortAsync


def loadonnx(args: Arguments) -> Runtime:
    return Onnxrt(args)

def loadhefasync(args: Arguments) -> RuntimeAsync:
    return HailortAsync(args)

def loadtflite(args: Arguments) -> Runtime:
    return Tflitert(args)

def loadmodel(args: Arguments) -> Union[Runtime, RuntimeAsync]:
    mp = Path(args.model_path)

    if (mp.suffix.lower() == ".onnx"):
        return loadonnx(args)
    
    elif (mp.suffix.lower() == ".tflite"):
        return loadtflite(args)
        
    elif (mp.suffix.lower() == ".hef"):
        return loadhefasync(args)
    
    raise ValueError(f"unsupport model type: {mp.suffix}")


import time

from typing import Optional

import cv2
import numpy


from sdk.data.inference_source import InferenceSource

class InferenceResult(InferenceSource):
    
    def __init__(
        self,
        instance: Optional[InferenceSource] = None,
	):
        
        if (instance is not None):
            super().__init__(
                image=instance.image,
                confidence=instance.confidence,
                threshold=instance.threshold,
                timestamp=instance.timestamp
            )
            
            self._spendtime: float = time.time() - self.timestamp
            
        else:
            super().__init__(
                image=numpy.empty((0, 0)),
                confidence=0,
                threshold=0,
                timestamp=0,
            )
            self._spendtime = 0
        
    @property
    def spendtime(self) -> float:
        return self._spendtime
    
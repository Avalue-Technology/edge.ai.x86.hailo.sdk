
import cv2

class InferenceSource():
    
    def __init__(
        self, 
        image: cv2.typing.MatLike,
        confidence: int,
        threshold: int,
        timestamp: float
    ):
        self._image = image
        self._confidence = confidence
        self._threshold = threshold
        self._timestamp = timestamp
        
    @property
    def image(self) -> cv2.typing.MatLike:
        return self._image
    
    @property
    def confidence(self) -> int:
        return self._confidence
    
    @property
    def threshold(self) -> int:
        return self._threshold
    
    @property
    def timestamp(self) -> float:
        return self._timestamp
    
    @timestamp.setter
    def timestamp(self, timestamp: float):
        self._timestamp = timestamp
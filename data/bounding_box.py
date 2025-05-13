

from dataclasses import dataclass
from typing import Self, Tuple

import cv2


@dataclass
class BoundingBox():
    
    id: int
    name: str
    confidence: float
    
    center: Tuple[int, int]
    width: int
    height: int
    
    
    
    def __init__(self):
        self.id = 0
        self.center = (0, 0)
        self.confidence = 0.0
    
    
    def lefttop(self) -> cv2.typing.Point:
        return (
            int(self.center[0] - self.width / 2),
            int(self.center[1] - self.height / 2)
        )
        
    def rightbottom(self) -> cv2.typing.Point:
        return (
            int(self.center[0] + self.width / 2),
            int(self.center[1] + self.height / 2)
        )
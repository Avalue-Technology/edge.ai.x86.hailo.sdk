

from dataclasses import dataclass, field


@dataclass
class ModelInformation:
    
    name: str
    device: str
    width: int
    height: int
    
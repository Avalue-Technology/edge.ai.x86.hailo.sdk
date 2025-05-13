

class InferenceResult():
    
    def __init__(
        self,
        spendtime: float,
        cpu_usage: float,
        image,
	):
        
        self._spendtime: float = spendtime
        self._cpu_usage: float = cpu_usage
        self._image = image
        
    @property
    def spendtime(self) -> float:
        return self._spendtime
    
    @property
    def cpu_usage(self) -> float:
        return self._cpu_usage
    
    
    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, image):
        self._iamge = image
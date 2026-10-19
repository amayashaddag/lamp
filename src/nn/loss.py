from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC): 
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float: 
        pass
    
    @abstractmethod
    def backward(self) -> np.ndarray: 
        pass
    
    
class MSELoss(Loss):
    y_pred: np.ndarray | None 
    y_true: np.ndarray | None 
    
    def __init__(self):
        self.y_pred = None 
        self.y_true = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float: 
        if y_pred.shape != y_true.shape: 
            raise ValueError(
            f"Shape mismatch: y_pred has shape {y_pred.shape}, "
            f"but y_true has shape {y_true.shape}"
        )
        loss = np.sum((y_pred - y_true) ** 2) / y_pred.size
        self.y_pred = y_pred 
        self.y_true = y_true 
        return loss 
    
    def backward(self) -> np.ndarray: 
        if self.y_pred is None or self.y_true is None: 
            raise ValueError(f"Expected forward method before backward")
        dL = (2 / self.y_pred.size) * (self.y_pred - self.y_true)
        return dL

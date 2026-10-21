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
    

class CrossEntropyLoss(Loss):
    y_pred: np.ndarray | None 
    y_true: np.ndarray | None
    
    def __init__(self):
            self.y_pred = None 
            self.y_true = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float: 
        self.y_true = y_true
        max_logits = y_pred.max(axis=1, keepdims=True)
        log_probs = y_pred - max_logits - np.log(np.exp(y_pred - max_logits).sum(axis=1, keepdims=True))
        self.y_pred = log_probs
        log_probs = - (log_probs[np.arange(len(y_true)), y_true])
        loss = (1 / log_probs.shape[0]) * log_probs.sum()
        return loss
    
    def backward(self) -> np.ndarray:
        if self.y_pred is None or self.y_true is None: 
            raise ValueError(f"Expected forward method before backward")
        ones = np.zeros_like(self.y_pred)
        ones[np.arange(self.y_pred.shape[0]), self.y_true] = 1
        dL = (1 / self.y_pred.shape[0]) * (np.exp(self.y_pred) - ones)
        return dL

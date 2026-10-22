import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray: 
        pass
    
    @abstractmethod
    def backward(self, dy: np.ndarray) -> np.ndarray:
        pass
    
    
class Linear(Layer): 
    in_features: int
    W: np.ndarray
    b: np.ndarray | None
    bias: bool 
    X: np.ndarray | None
    dW: np.ndarray
    db: np.ndarray | None
    
    def __init__(self, in_features: int, out_features: int, bias: bool):
        self.in_features = in_features
        self.W = np.random.normal(0, np.sqrt(2 / (in_features + out_features)), size=(in_features, out_features))
        self.b = np.zeros(out_features) if bias else None
        self.bias = bias
        self.X = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        
    def forward(self, X: np.ndarray) -> np.ndarray: 
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array (batch_size, in_features), got shape {X.shape}")
        if X.shape[1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {X.shape[1]}")
        y = X @ self.W
        if self.bias: 
            y = y + self.b
        self.X = X
        return y
    
    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.X is None: 
            raise ValueError(f"Expected forward method before backward")
        if dy.shape != (self.X.shape[0], self.W.shape[1]): 
            raise ValueError(
                f"Expected dy shape {(self.X.shape[0], self.W.shape[1])}, "
                f"got {dy.shape}"
            )
        dW = self.X.T @ dy
        self.dW[...] = dW
        if self.bias and self.db is not None:
            db = np.sum(dy, axis=0)
            self.db[...] = db
        dX = dy @ self.W.T 
        return dX
        

class ReLU(Layer):
    X: np.ndarray | None
    
    def __init__(self): 
        self.X = None 
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        y = np.maximum(X, 0)
        self.X = X 
        return y
    
    def backward(self, dy: np.ndarray) -> np.ndarray: 
        if self.X is None: 
            raise ValueError(f"Expected forward method before backward")
        dX = dy * (self.X > 0)
        return dX
    

class Sigmoid(Layer): 
    X: np.ndarray | None 
    
    def __init__(self): 
        self.X = None 
        
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
        
    def forward(self, X: np.ndarray) -> np.ndarray: 
        y = Sigmoid.sigmoid(X)
        self.X = y
        return y
    
    def backward(self, dy: np.ndarray) -> np.ndarray: 
        if self.X is None: 
            raise ValueError(f"Expected forward method before backward")
        dX = dy * (self.X * (1 - self.X))
        return dX
        
        
class Flatten(Layer):
    X: np.ndarray | None

    def __init__(self):
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        if X.ndim < 2:
            return X.copy()
        return X.reshape(X.shape[0], -1)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.X is None:
            raise RuntimeError("forward must be called before backward")
        return dy.reshape(self.X.shape)
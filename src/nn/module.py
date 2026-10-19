from nn.layer import Layer, Linear
import numpy as np


class Module():
    layers: list[Layer]
    
    def __init__(self, layers: list[Layer]): 
        self.layers = layers  
    
    def forward(self, X: np.ndarray) -> np.ndarray: 
        y = X
        for l in self.layers: 
            y = l.forward(y)
        return y
        
    def backward(self, dy: np.ndarray) -> np.ndarray: 
        for l in reversed(self.layers): 
            dy = l.backward(dy)
        return dy
    
    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = []
        param_layers: list[Linear] = [l for l in self.layers if isinstance(l, Linear)]
        for l in param_layers: 
            params.append((l.W, l.dW))
            if l.bias: 
                params.append((l.b, l.db))
        return params        
        
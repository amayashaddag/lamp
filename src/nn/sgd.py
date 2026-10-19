import numpy as np

class SGD: 
    parameters: list[tuple[np.ndarray, np.ndarray]]
    lr: float 
    
    def __init__(self, parameters: list[tuple[np.ndarray, np.ndarray]], lr: float): 
        self.parameters = parameters
        self.lr = lr 
        
    def step(self): 
        for param, grad in self.parameters:
            if grad is None: 
                continue
            param -= self.lr * grad 
        
    def zero_grad(self): 
        for _, grad in self.parameters: 
            if grad is None: 
                continue
            grad.fill(0.0)
        
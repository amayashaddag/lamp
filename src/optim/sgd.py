import numpy as np

class SGD: 
    parameters: list[tuple[np.ndarray, np.ndarray]]
    momentum: float
    velocities: list[np.ndarray]
    lr: float 
    
    def __init__(self, parameters: list[tuple[np.ndarray, np.ndarray]], lr: float, momentum: float): 
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        self.parameters = parameters
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p, _ in parameters]
        self.lr = lr 
        
    def step(self): 
        for (param, grad), vel in zip(self.parameters, self.velocities):
            if grad is None: 
                continue
            vel[...] = self.momentum * vel + grad
            param -= self.lr * vel 
        
    def zero_grad(self): 
        for _, grad in self.parameters: 
            if grad is None: 
                continue
            grad.fill(0.0)
        
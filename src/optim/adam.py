import numpy as np


class Adam(): 
    means: list[np.ndarray]
    velocities: list[np.ndarray]
    beta_1: float
    beta_2: float
    parameters: list[tuple[np.ndarray, np.ndarray]]
    t: int
    lr: float
    eps: float
    
    def __init__(
        self, 
        parameters: list[tuple[np.ndarray, np.ndarray]],
        beta_1: float,
        beta_2: float,
        lr: float,
        eps: float,
    ): 
        self.parameters = parameters
        self.beta_1 = beta_1
        self.beta_2 = beta_2 
        self.means = [np.zeros_like(p) for p, _ in parameters]
        self.velocities = [np.zeros_like(p) for p, _ in parameters]
        self.t = 0
        self.lr = lr
        self.eps = eps
        
    def step(self): 
        self.t += 1
        for (param, grad), mean, vel in zip(self.parameters, self.means, self.velocities): 
            if grad is None: 
                continue
            mean[...] = (self.beta_1 * mean + (1 - self.beta_1) * grad)
            unbiased_mean = mean / (1 - self.beta_1 ** self.t)
            vel[...] = (self.beta_2 * vel + (1 - self.beta_2) * grad ** 2)
            unbiased_vel = vel / (1 - self.beta_2 ** self.t)
            param -= self.lr * unbiased_mean / (np.sqrt(unbiased_vel) + self.eps)
            
    def zero_grad(self): 
        for _, grad in self.parameters: 
            if grad is None: 
                continue
            grad.fill(0.0)
            
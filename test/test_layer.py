import numpy as np
from nn.layer import Flatten


def test_flatten():
    b, h, w, x = 5, 5, 6, 3
    X = np.random.randn(b, h, w, x)
    flatten = Flatten()
    y = flatten.forward(X)
    assert len(y.shape) == 2
    assert y.shape[0] == b
    assert y.shape[1] == h * w * x
    

def test_restore(): 
    b, h, w, x = 5, 5, 6, 3
    X = np.random.randn(b, h, w, x)
    flatten = Flatten()
    y = flatten.forward(X)
    X_restored = flatten.backward(y)
    assert X_restored.shape == X.shape
    
    
    
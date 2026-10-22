from optim.sgd import SGD

import numpy as np


def test_sgd_update(): 
    np.random.seed(42)
    nb_params = 5
    params = []
    params_backup = []
    for _ in range(nb_params): 
        p = np.random.randn(5)
        g = np.random.randn(5)
        params.append((p, g))
        params_backup.append((p.copy(), g.copy()))
    lr = 1e-3
    optimizer = SGD(parameters=params, lr=lr, momentum=0.0)
    optimizer.step()
    for (p_old, g), (p, _) in zip(params_backup, params): 
        np.testing.assert_allclose(p, p_old - lr * g, rtol=1e-4, atol=1e-6)
        

def test_sgd_momentum(): 
    p, g = np.random.randn(3, 3), np.random.randn(3, 3)
    params = [(p, g)]
    params_backup = [(p.copy(), g.copy())]
    p_old, _ = params_backup[0]
    lr = 1e-3
    momentum = 0.9
    optimizer = SGD(parameters=params, lr=lr, momentum=momentum)
    optimizer.step()
    np.testing.assert_allclose(p, p_old - lr * g, rtol=1e-4, atol=1e-6)
    p_old[...] = p
    optimizer.step()
    np.testing.assert_allclose(p, p_old - lr * (momentum * g + g))
    g_new = np.random.randn(3, 3)
    params[0] = (p, g_new)
    p_old[...] = p
    optimizer.step()
    np.testing.assert_allclose(p, p_old - lr * (momentum * (momentum * g + g) + g_new), rtol=1e-4, atol=1e-6)
    

def test_zero_grad(): 
    np.random.seed(42)
    nb_params = 3
    params = []
    for _ in range(nb_params): 
        p = np.random.randn(5)
        g = np.random.randn(5)
        params.append((p, g))
    optimizer = SGD(parameters=params, lr=1e-3, momentum=0.9)
    optimizer.zero_grad()
    for _, g in params: 
        np.testing.assert_equal(g, np.zeros_like(g))
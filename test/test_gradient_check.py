from nn.layer import Linear, ReLU, Sigmoid
from nn.loss import MSELoss, CrossEntropyLoss

import numpy as np


def test_mse_linear_gradient(): 
    np.random.seed(42)
    X = np.random.randn(4, 3)
    y_true = np.random.randn(4, 2)
    layer = Linear(3, 2, bias=True)
    assert layer.b is not None
    loss_fn = MSELoss()
    y_pred = layer.forward(X)
    loss_fn.forward(y_pred, y_true)
    dy = loss_fn.backward()
    layer.backward(dy)
    assert layer.db is not None
    dW_analytic = layer.dW.copy()
    db_analytic = layer.db.copy()
    eps = 1e-5
    dW_numeric = np.zeros_like(layer.W)
    db_numeric = np.zeros_like(layer.b)
    for idx in np.ndindex(layer.W.shape):
        original = layer.W[idx]
        layer.W[idx] = original + eps
        loss_plus = loss_fn.forward(layer.forward(X), y_true)
        layer.W[idx] = original - eps
        loss_minus = loss_fn.forward(layer.forward(X), y_true)
        layer.W[idx] = original
        dW_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    for idx in np.ndindex(layer.b.shape):
        original = layer.b[idx]
        layer.b[idx] = original + eps
        loss_plus = loss_fn.forward(layer.forward(X), y_true)
        layer.b[idx] = original - eps
        loss_minus = loss_fn.forward(layer.forward(X), y_true)
        layer.b[idx] = original
        db_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dW_analytic, dW_numeric, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(db_analytic, db_numeric, rtol=1e-4, atol=1e-6)
    

def test_cross_entropy_linear_gradient():
    np.random.seed(42)
    X = np.random.randn(4, 3)
    y_true = np.array([0, 2, 1, 0])
    layer = Linear(3, 3, bias=True)
    assert layer.b is not None
    loss_fn = CrossEntropyLoss()
    y_pred = layer.forward(X)
    loss_fn.forward(y_pred, y_true)
    dy = loss_fn.backward()
    layer.backward(dy)
    assert layer.db is not None
    dW_analytic = layer.dW.copy()
    db_analytic = layer.db.copy()
    eps = 1e-5
    dW_numeric = np.zeros_like(layer.W)
    db_numeric = np.zeros_like(layer.b)
    for idx in np.ndindex(layer.W.shape):
        original = layer.W[idx]
        layer.W[idx] = original + eps
        loss_plus = loss_fn.forward(layer.forward(X), y_true)
        layer.W[idx] = original - eps
        loss_minus = loss_fn.forward(layer.forward(X), y_true)
        layer.W[idx] = original
        dW_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    for idx in np.ndindex(layer.b.shape):
        original = layer.b[idx]
        layer.b[idx] = original + eps
        loss_plus = loss_fn.forward(layer.forward(X), y_true)
        layer.b[idx] = original - eps
        loss_minus = loss_fn.forward(layer.forward(X), y_true)
        layer.b[idx] = original
        db_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dW_analytic, dW_numeric, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(db_analytic, db_numeric, rtol=1e-4, atol=1e-6)


def test_relu_gradients():
    X = np.array([[-2.0, -0.5, 0.5, 2.0]])
    dy = np.array([[1.0, 2.0, 3.0, 4.0]])
    layer = ReLU()
    layer.forward(X)
    dx_analytic = layer.backward(dy)
    eps = 1e-5
    dx_numeric = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        original = X[idx]
        X[idx] = original + eps
        loss_plus = np.sum(layer.forward(X) * dy)
        X[idx] = original - eps
        loss_minus = np.sum(layer.forward(X) * dy)
        X[idx] = original
        dx_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-4, atol=1e-6)
    

def test_sigmoid_gradients():
    X = np.array([[-2.0, -0.5, 0.5, 2.0]])
    dy = np.array([[1.0, 2.0, 3.0, 4.0]])
    layer = Sigmoid()
    layer.forward(X)
    dx_analytic = layer.backward(dy)
    eps = 1e-5
    dx_numeric = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        original = X[idx]
        X[idx] = original + eps
        loss_plus = np.sum(layer.forward(X) * dy)
        X[idx] = original - eps
        loss_minus = np.sum(layer.forward(X) * dy)
        X[idx] = original
        dx_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-4, atol=1e-6)

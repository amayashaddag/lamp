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
    dX_analytic = layer.backward(dy)
    assert layer.db is not None
    dW_analytic = layer.dW.copy()
    db_analytic = layer.db.copy()
    eps = 1e-5
    dX_numeric = np.zeros_like(X)
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
    for idx in np.ndindex(X.shape):
        original = X[idx]
        X[idx] = original + eps
        loss_plus = loss_fn.forward(layer.forward(X), y_true)
        X[idx] = original - eps
        loss_minus = loss_fn.forward(layer.forward(X), y_true)
        X[idx] = original
        dX_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dW_analytic, dW_numeric, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(db_analytic, db_numeric, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(dX_analytic, dX_numeric, rtol=1e-4, atol=1e-6)
    

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
    dX_analytic = layer.backward(dy)
    assert layer.db is not None
    dW_analytic = layer.dW.copy()
    db_analytic = layer.db.copy()
    eps = 1e-5
    dW_numeric = np.zeros_like(layer.W)
    db_numeric = np.zeros_like(layer.b)
    dX_numeric = np.zeros_like(X)
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
    for idx in np.ndindex(X.shape):
        original = X[idx]
        X[idx] = original + eps
        loss_plus = loss_fn.forward(layer.forward(X), y_true)
        X[idx] = original - eps
        loss_minus = loss_fn.forward(layer.forward(X), y_true)
        X[idx] = original
        dX_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dW_analytic, dW_numeric, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(db_analytic, db_numeric, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(dX_analytic, dX_numeric, rtol=1e-4, atol=1e-6)


def test_relu_gradients():
    X = np.array([[-2.0, -0.5, 0.5, 2.0]])
    dy = np.array([[1.0, 2.0, 3.0, 4.0]])
    layer = ReLU()
    layer.forward(X)
    dX_analytic = layer.backward(dy)
    eps = 1e-5
    dX_numeric = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        original = X[idx]
        X[idx] = original + eps
        loss_plus = np.sum(layer.forward(X) * dy)
        X[idx] = original - eps
        loss_minus = np.sum(layer.forward(X) * dy)
        X[idx] = original
        dX_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dX_analytic, dX_numeric, rtol=1e-4, atol=1e-6)
    

def test_sigmoid_gradients():
    X = np.array([[-2.0, -0.5, 0.5, 2.0]])
    dy = np.array([[1.0, 2.0, 3.0, 4.0]])
    layer = Sigmoid()
    layer.forward(X)
    dX_analytic = layer.backward(dy)
    eps = 1e-5
    dX_numeric = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        original = X[idx]
        X[idx] = original + eps
        loss_plus = np.sum(layer.forward(X) * dy)
        X[idx] = original - eps
        loss_minus = np.sum(layer.forward(X) * dy)
        X[idx] = original
        dX_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dX_analytic, dX_numeric, rtol=1e-4, atol=1e-6)


def test_mse_gradients(): 
    np.random.seed(42)
    y_true = np.random.randn(4, 2)
    y_pred = np.random.randn(4, 2)
    loss_fn = MSELoss()
    loss_fn.forward(y_pred, y_true)
    dy_analytic = loss_fn.backward()
    dy_numeric = np.zeros_like(y_pred)
    eps = 1e-5
    for idx in np.ndindex(y_pred.shape):
        original = y_pred[idx]
        y_pred[idx] = original + eps
        loss_plus = loss_fn.forward(y_pred, y_true)
        y_pred[idx] = original - eps
        loss_minus = loss_fn.forward(y_pred, y_true)
        y_pred[idx] = original
        dy_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dy_analytic, dy_numeric, rtol=1e-4, atol=1e-6)
    

def test_cross_entropy_gradients(): 
    np.random.seed(42)
    y_true = np.random.randint(0, 4, size=4)
    y_pred = np.random.randn(4, 4)
    loss_fn = CrossEntropyLoss()
    loss_fn.forward(y_pred, y_true)
    dy_analytic = loss_fn.backward()
    dy_numeric = np.zeros_like(y_pred)
    eps = 1e-5
    for idx in np.ndindex(y_pred.shape):
        original = y_pred[idx]
        y_pred[idx] = original + eps
        loss_plus = loss_fn.forward(y_pred, y_true)
        y_pred[idx] = original - eps
        loss_minus = loss_fn.forward(y_pred, y_true)
        y_pred[idx] = original
        dy_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
    np.testing.assert_allclose(dy_analytic, dy_numeric, rtol=1e-4, atol=1e-6)
#%%

from nn.module import Module 
from nn.layer import Linear, Sigmoid, ReLU
from optim.sgd import SGD
from nn.loss import MSELoss

import numpy as np

#%%

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y_true = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])
model = Module([
    Linear(2, 8, bias=True), 
    ReLU(),
    Linear(8, 1, bias=True),
    Sigmoid(),
])
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)
for epoch in range(5000):
    y_pred = model.forward(X)
    loss = loss_fn.forward(y_pred, y_true)
    optimizer.zero_grad()
    dy = loss_fn.backward()
    model.backward(dy)
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, loss={loss:.6f}")
        
#%%

y_pred = model.forward(X)
y_binary = (y_pred >= 0.5).astype(int)
print("Real predictions :")
print(y_pred)
print("\nBinary predictions :")
print(y_binary)
print("\nExpected values :")
print(y_true.astype(int))
accuracy = np.mean(y_binary == y_true) * 100
print(f"\nAccuracy : {accuracy:.1f}%")
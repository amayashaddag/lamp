#%%
import numpy as np
import kagglehub
import struct
import matplotlib.pyplot as plt
from pathlib import Path

from nn.module import Module 
from nn.layer import Linear, ReLU
from nn.sgd import SGD
from nn.loss import CrossEntropyLoss

#%%

path = Path(kagglehub.dataset_download("hojjatk/mnist-dataset"))

#%%

def load_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        _, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(n_images, rows * cols).astype(np.float32) / 255.0


def load_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        _, n_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)

#%%

X_train = load_images(path / "train-images.idx3-ubyte")
y_train = load_labels(path / "train-labels.idx1-ubyte")
X_test = load_images(path / "t10k-images.idx3-ubyte")
y_test = load_labels(path / "t10k-labels.idx1-ubyte")

#%%

model = Module([
    Linear(784, 128, bias=True),
    ReLU(),
    Linear(128, 10, bias=True),
])
loss_fn = CrossEntropyLoss()
optim = SGD(model.parameters(), lr=0.1)
epochs = 10
batchs = X_train.shape[0] // 64
for epoch in range(epochs):
    i = 0
    loss = 0.0
    for batch in range(batchs): 
        X = X_train[i : min(i + 64, X_train.shape[0])]
        y_true = y_train[i : min(i + 64, y_train.shape[0])]
        y_pred = model.forward(X)
        loss = loss_fn.forward(y_pred, y_true)
        optim.zero_grad()
        dy = loss_fn.backward()
        model.backward(dy)
        optim.step()
        i += 64
    print(f"Epoch {epoch}, loss={loss:.6f}")
    
#%%

y_pred = model.forward(X_test)
test_loss = loss_fn.forward(y_pred, y_test)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = np.mean(y_pred_classes == y_test)
print(f"Test loss: {test_loss:.6f}")
print(f"Test accuracy: {accuracy:.2%}")

#%% 

ind = 76
image = X_test[ind].reshape(28, 28)
plt.imshow(image)
plt.show()
y_pred = model.forward(X_test[ind].reshape(1, 784))
y_pred_class = np.argmax(y_pred, axis=1)
print(f"Predicted value: {y_pred_class[0]}")
print(f"Real value: {y_test[ind]}")
"""預訓練 64→10→10→2 DNN（8x8 手寫數字 0/1 二分類），輸出 weights.json 供 index.html 嵌入。"""
import json

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
rng = np.random.default_rng(SEED)

# 資料：sklearn digits 本身就是 8x8、像素 0-16，只取 0 與 1
digits = load_digits()
mask = digits.target < 2
X = digits.data[mask] / 16.0
y = digits.target[mask]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"樣本數：train={len(X_train)}, test={len(X_test)}")

# 初始化 64→10→10→2
def init(fan_in, fan_out):
    return rng.normal(0, np.sqrt(1.0 / fan_in), (fan_in, fan_out))

W1, b1 = init(64, 10), np.zeros(10)
W2, b2 = init(10, 10), np.zeros(10)
W3, b3 = init(10, 2), np.zeros(2)

Y = np.eye(2)[y_train]  # one-hot
lr = 0.5
for epoch in range(2000):
    # forward
    h1 = np.tanh(X_train @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    z = h2 @ W3 + b3
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(Y * np.log(p + 1e-12), axis=1))

    # backward
    n = len(X_train)
    dz = (p - Y) / n
    dW3, db3 = h2.T @ dz, dz.sum(axis=0)
    dh2 = dz @ W3.T * (1 - h2**2)
    dW2, db2 = h1.T @ dh2, dh2.sum(axis=0)
    dh1 = dh2 @ W2.T * (1 - h1**2)
    dW1, db1 = X_train.T @ dh1, dh1.sum(axis=0)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2
    W3 -= lr * dW3; b3 -= lr * db3

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss={loss:.4f}")

def forward(X):
    h2 = np.tanh(np.tanh(X @ W1 + b1) @ W2 + b2)
    z = h2 @ W3 + b3
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)

p_test = forward(X_test)
acc = (p_test.argmax(axis=1) == y_test).mean()
test_loss = -np.mean(np.log(p_test[np.arange(len(y_test)), y_test] + 1e-12))
print(f"test accuracy = {acc:.4f}, test loss = {test_loss:.4f}")
print(f"W3 範圍：min={W3.min():.3f}, max={W3.max():.3f}")
print(f"b3 = {b3}")

out = {
    "W1": np.round(W1, 5).tolist(), "b1": np.round(b1, 5).tolist(),
    "W2": np.round(W2, 5).tolist(), "b2": np.round(b2, 5).tolist(),
    "W3_trained": np.round(W3, 5).tolist(), "b3": np.round(b3, 5).tolist(),
    # 測試集以 8x8 整數 (0-16) 儲存，前端再除以 16
    "X_test": (X_test * 16).astype(int).tolist(),
    "y_test": y_test.tolist(),
}
with open("weights.json", "w") as f:
    json.dump(out, f)
print("已輸出 weights.json")

# FashionMNIST Softmax 迴歸分類器 — 說明文件

> 對應程式碼：[`fashion_mnist.py`](./fashion_mnist.py)

---

## 目錄

1. [任務概覽](#1-任務概覽)
2. [資料集介紹：FashionMNIST](#2-資料集介紹fashionmnist)
3. [模型：Softmax 迴歸](#3-模型softmax-迴歸)
4. [數學原理](#4-數學原理)
5. [程式碼架構](#5-程式碼架構)
6. [逐步說明](#6-逐步說明)
   - [6.1 GPU 裝置設定與參數初始化](#61-gpu-裝置設定與參數初始化)
   - [6.2 標籤對照：`get_fashion_mnist_labels`](#62-標籤對照get_fashion_mnist_labels)
   - [6.3 圖像顯示：`show_images`](#63-圖像顯示show_images)
   - [6.4 Softmax 函數：`softmax`](#64-softmax-函數softmax)
   - [6.5 網路前向傳播：`net`](#65-網路前向傳播net)
   - [6.6 損失函數：`cross_entropy`](#66-損失函數cross_entropy)
   - [6.7 準確率計算：`accuracy`](#67-準確率計算accuracy)
   - [6.8 優化器：`SGD`](#68-優化器sgd)
   - [6.9 資料載入](#69-資料載入)
   - [6.10 訓練迴圈](#610-訓練迴圈)
7. [超參數說明](#7-超參數說明)
8. [與線性迴歸的差異對比](#8-與線性迴歸的差異對比)
9. [預期執行結果](#9-預期執行結果)
10. [常見問題](#10-常見問題)

---

## 1. 任務概覽

本程式將 FashionMNIST 服裝圖像分為 **10 個類別**，使用**從零實作的 Softmax 迴歸**，不依賴 `nn.Linear` 或 `nn.Softmax` 等高階 API，完整展示多元分類的核心流程。

| 項目 | 內容 |
|------|------|
| 任務類型 | 多元分類（10 類） |
| 輸入 | 28×28 灰階圖像，展平為 784 維向量 |
| 輸出 | 10 個類別的機率分佈 |
| 模型 | Softmax 迴歸（單層神經網路） |
| 損失函數 | 交叉熵（Cross Entropy） |
| 優化器 | 小批次 SGD |

---

## 2. 資料集介紹：FashionMNIST

FashionMNIST 是 MNIST 手寫數字資料集的**服裝版本替代品**，共 10 個類別：

| 標籤 | 類別 | 標籤 | 類別 |
|------|------|------|------|
| 0 | t-shirt（T恤） | 5 | sandal（涼鞋） |
| 1 | trouser（長褲） | 6 | shirt（襯衫） |
| 2 | pullover（套頭衫） | 7 | sneaker（運動鞋） |
| 3 | dress（洋裝） | 8 | bag（包包） |
| 4 | coat（外套） | 9 | ankle boot（短靴） |

- **訓練集**：60,000 張
- **測試集**：10,000 張
- **圖像尺寸**：28 × 28 像素，灰階（1 個 channel）

---

## 3. 模型：Softmax 迴歸

Softmax 迴歸是**多元邏輯斯迴歸的推廣**，也是最簡單的神經網路分類器（單層）：

```
輸入圖像 (28×28) → 展平 (784,) → 線性變換 (784→10) → Softmax → 機率分佈 (10,)
```

雖然稱為「迴歸」，但它的輸出是各類別的**機率**，因此本質上是分類模型。

---

## 4. 數學原理

### Softmax 函數

將原始分數（logits）轉換為機率分佈，確保輸出值介於 0~1 且總和為 1：

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

### 交叉熵損失

衡量預測機率分佈與真實標籤的差距：

```
L(ŷ, y) = -log(ŷ_y)
```

其中 `ŷ_y` 是正確類別的預測機率。直覺上：若正確類別的預測機率越高，損失越小。

### 整體前向傳播

```
O = X · W + b          (線性變換，shape: (batch, 10))
ŷ = softmax(O)         (轉換為機率，shape: (batch, 10))
L = -log(ŷ[y])        (只取正確類別的機率)
```

---

## 5. 程式碼架構

```
fashion_mnist.py
│
├── [裝置設定]      device 選擇（CUDA / CPU）
├── [參數初始化]    W (784×10), b (10,)
├── [函式定義]
│   ├── get_fashion_mnist_labels()  — 數字標籤 → 文字
│   ├── show_images()               — 圖像網格視覺化
│   ├── softmax()                   — Softmax 激活函數
│   ├── net()                       — 前向傳播（展平 + 線性 + softmax）
│   ├── cross_entropy()             — 交叉熵損失
│   ├── accuracy()                  — 批次準確率計算
│   ├── SGD()                       — 小批次梯度下降
│   └── data_iter()                 — 手動小批次迭代器（保留備用）
├── [資料載入]      FashionMNIST + DataLoader
└── [訓練迴圈]      前向 → 損失 → 反向 → 更新 → 評估
```

---

## 6. 逐步說明

### 6.1 GPU 裝置設定與參數初始化

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_input = 28 * 28   # 784
num_output = 10       # 10 個類別

W = torch.normal(0, 0.01, size=(num_input, num_output), requires_grad=True, device=device)
b = torch.zeros(num_output, requires_grad=True, device=device)
```

- **`W` 形狀 `(784, 10)`**：每一列對應一個類別的權重向量。
- **`W` 用小隨機值初始化**（`torch.normal(0, 0.01, ...)`）：避免對稱性問題，使各神經元在訓練初期學習不同的特徵。
- **`b` 初始化為 0**：偏置對稱性問題影響較小。
- **`device=device`**：在 GPU 上直接建立，避免額外的搬移開銷。

> **重要**：`W`、`b` 與訓練資料必須在同一個裝置上，否則會拋出 `RuntimeError: Expected all tensors to be on the same device`。

### 6.2 標籤對照：`get_fashion_mnist_labels`

```python
def get_fashion_mnist_labels(labels):
    text_labels = [
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot"
    ]
    return [text_labels[int(label)] for label in labels]
```

將整數標籤（0~9）轉換為對應的文字類別名稱，供圖像視覺化時顯示標題使用。

### 6.3 圖像顯示：`show_images`

```python
def show_images(imgs, num_rows, num_cols, titles=None):
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.squeeze().numpy()  # (1,28,28) → (28,28)
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
```

關鍵細節：
- **`.squeeze()`**：移除大小為 1 的 channel 維度。`ToTensor()` 後圖像形狀為 `(1, 28, 28)`，但 `matplotlib.imshow` 的灰階圖像要求 `(H, W) = (28, 28)`。
- **關閉座標軸**：圖像不需要顯示座標刻度。

### 6.4 Softmax 函數：`softmax`

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
```

- `X` 形狀：`(batch_size, 10)`
- `X_exp.sum(1, keepdim=True)` 對每一列（每筆資料的 10 個類別分數）加總，結果形狀 `(batch_size, 1)`。
- `keepdim=True` 保持維度，讓除法能正確廣播：`(batch, 10) / (batch, 1)` → `(batch, 10)`。

> **注意**：此實作為教學用，實際使用中 `torch.softmax` 有數值穩定性優化（減去最大值再做 exp），可避免 overflow 問題。

### 6.5 網路前向傳播：`net`

```python
def net(X):
    return softmax(torch.matmul(X.reshape(-1, num_input), W) + b)
```

完整流程：
1. `X.reshape(-1, num_input)`：將圖像從 `(batch, 1, 28, 28)` 展平為 `(batch, 784)`。
2. `torch.matmul(..., W) + b`：線性變換，輸出形狀 `(batch, 10)`。
3. `softmax(...)`：轉換為機率分佈，輸出形狀 `(batch, 10)`，每列加總為 1。

### 6.6 損失函數：`cross_entropy`

```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y])
```

- `y_hat[range(len(y)), y]`：進階索引（fancy indexing），從每筆資料的 10 個預測機率中，只取出**正確類別**的機率值。
- 例如：`y = [3, 7, 0]`，則取 `y_hat[0, 3]`、`y_hat[1, 7]`、`y_hat[2, 0]`。
- `- torch.log(...)`：機率越高，損失越小。

### 6.7 準確率計算：`accuracy`

```python
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / y.numel()
```

- `d2l.argmax(y_hat, dim=1)`：取每筆資料預測機率最高的類別索引。
- `y_hat.type(y.dtype)`：確保比較時資料型別一致。
- 回傳值為 **[0, 1] 之間的浮點數**，代表這個批次的預測準確率。

### 6.8 優化器：`SGD`

```python
def SGD(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

與線性迴歸中的 `sgd` 完全相同的邏輯，詳見 `linear_regression.md` 的 Q5 說明。

### 6.9 資料載入

```python
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True,  transform=trans, download=True)
mnist_test  = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader  = data.DataLoader(mnist_test,  batch_size=batch_size, shuffle=False)
```

- **`transforms.ToTensor()`**：將 PIL 圖像（像素值 0~255）轉換為 PyTorch 張量（值域 0.0~1.0），並從 `(H, W, C)` 轉置為 `(C, H, W)` 格式（channel-first）。
- **`DataLoader`**：自動處理批次分割、多執行緒載入、資料打亂（shuffle）。
  - 訓練集設 `shuffle=True`，每個 epoch 資料順序不同，增加訓練多樣性。
  - 測試集設 `shuffle=False`，評估時保持固定順序。

### 6.10 訓練迴圈

```python
for epoch in range(EPOCH):
    # --- Training ---
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        X = X.reshape(-1, num_input)     # (batch,1,28,28) → (batch,784)
        y_hat = net(X)                   # 前向傳播
        loss = cross_entropy(y_hat, y)   # 計算損失
        loss.sum().backward()            # 反向傳播
        SGD([W, b], lr, batch_size)      # 更新參數
        train_loss_sum += loss.sum().item()
        train_acc_sum  += accuracy(y_hat, y) * len(y)
        n += len(y)

    # --- Evaluation ---
    test_acc_sum, m = 0.0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            X = X.reshape(-1, num_input)
            test_acc_sum += accuracy(net(X), y) * len(y)
            m += len(y)

    print(f'epoch {epoch+1:2d}  loss {train_loss_sum/n:.4f}  '
          f'train acc {train_acc_sum/n:.3f}  test acc {test_acc_sum/m:.3f}')
```

每個 epoch 流程圖：

```
┌──────────────────────────────────────────────────────────────┐
│  for each mini-batch (訓練):                                 │
│                                                              │
│  [1] 搬移至 GPU  X, y = X.to(device), y.to(device)          │
│  [2] 展平圖像    X = X.reshape(-1, 784)                      │
│  [3] 前向傳播   y_hat = softmax(X·W + b)   shape: (B, 10)   │
│  [4] 損失計算   L = -log(ŷ_y)              shape: (B,)      │
│  [5] 反向傳播   loss.sum().backward()                        │
│  [6] 參數更新   W -= lr * W.grad / batch_size                │
│                 b -= lr * b.grad / batch_size                │
│                                                              │
│  epoch 結束後（評估，no_grad）:                              │
│  [7] 在整個測試集上計算準確率並印出                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. 超參數說明

| 超參數 | 值 | 說明 | 調整建議 |
|--------|-----|------|----------|
| `batch_size` | `2048` | 批次大小 | 可嘗試 `256`~`4096`；GPU 記憶體允許時大批次更快 |
| `lr` | `0.1` | 學習率 | Softmax 迴歸通常需要比線性迴歸更大的學習率 |
| `EPOCH` | `20` | 訓練輪數 | Softmax 迴歸在此資料集上約 5~10 個 epoch 即可收斂 |

---

## 8. 與線性迴歸的差異對比

| 面向 | 線性迴歸 (`linear_regression.py`) | Softmax 分類 (`fashion_mnist.py`) |
|------|----------------------------------|-----------------------------------|
| 任務 | 預測連續數值 | 預測離散類別 |
| 輸出層 | 純線性 `X·w + b`，1 個輸出 | softmax，10 個輸出（機率） |
| 損失函數 | 均方誤差 `½(ŷ-y)²` | 交叉熵 `-log(ŷ_y)` |
| 評估指標 | 損失值（越小越好） | 準確率（越高越好） |
| 資料來源 | `d2l.synthetic_data` 合成資料 | FashionMNIST 真實圖像資料集 |
| 資料載入 | 手動 `data_iter` 生成器 | PyTorch `DataLoader` |
| 參數維度 | `w: (2,)`, `b: (1,)` | `W: (784, 10)`, `b: (10,)` |

---

## 9. 預期執行結果

訓練成功後，終端輸出類似：

```
60000 10000
epoch  1  loss 1.3456  train acc 0.625  test acc 0.722
epoch  2  loss 0.9834  train acc 0.733  test acc 0.762
epoch  5  loss 0.7621  train acc 0.778  test acc 0.793
epoch 10  loss 0.6512  train acc 0.802  test acc 0.812
epoch 20  loss 0.5943  train acc 0.821  test acc 0.825
```

- Softmax 迴歸在 FashionMNIST 上的典型最終準確率約 **82%~85%**。
- 若要進一步提升，需改用多層神經網路（MLP）或捲積神經網路（CNN）。

---

## 10. 常見問題

**Q1：為什麼用 `cross_entropy` 而不是 `squared_loss`？**

均方誤差對機率分佈的梯度計算效果差，容易飽和（gradient vanishing）。交叉熵配合 Softmax 的梯度公式非常簡潔：`∂L/∂o_i = ŷ_i - y_i`（`y_i` 為 one-hot 值），訓練更高效。

---

**Q2：`X.reshape(-1, num_input)` 中的 `-1` 是什麼意思？**

`-1` 代表讓 PyTorch 自動推算這個維度的大小。`reshape(-1, 784)` 等同於：「無論批次大小是多少，都把每張圖像展平成 784 維向量。」

---

**Q3：訓練集 shuffle=True，測試集 shuffle=False，為什麼？**

訓練時打亂資料可避免模型記住資料出現的順序，使梯度更新更具代表性。測試集是固定的評估基準，不需要打亂（且打亂後結果相同）。

---

**Q4：`d2l.argmax` 與 `torch.argmax` 有什麼差別？**

`d2l.argmax` 是 d2l 函式庫對 `torch.argmax` 的薄包裝，功能相同，皆回傳指定維度上的最大值索引。

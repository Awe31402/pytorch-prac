# 線性迴歸 — PyTorch 從零實作說明文件

> 對應程式碼：[`linear_regression.py`](./linear_regression.py)

---

## 目錄

1. [概念概覽](#1-概念概覽)
2. [數學原理](#2-數學原理)
3. [程式碼架構](#3-程式碼架構)
4. [逐步說明](#4-逐步說明)
   - [4.1 環境設定與 GPU 支援](#41-環境設定與-gpu-支援)
   - [4.2 合成資料集生成](#42-合成資料集生成)
   - [4.3 資料迭代器 `data_iter`](#43-資料迭代器-data_iter)
   - [4.4 線性迴歸模型 `linreg`](#44-線性迴歸模型-linreg)
   - [4.5 損失函數 `squared_loss`](#45-損失函數-squared_loss)
   - [4.6 優化器 `sgd`](#46-優化器-sgd)
   - [4.7 參數初始化](#47-參數初始化)
   - [4.8 訓練迴圈](#48-訓練迴圈)
5. [超參數說明](#5-超參數說明)
6. [GPU 加速原理](#6-gpu-加速原理)
7. [執行結果與收斂](#7-執行結果與收斂)
8. [常見問題](#8-常見問題)

---

## 1. 概念概覽

本程式以 **純 PyTorch**（不使用任何高階 API 如 `nn.Linear`）實作線性迴歸，目的是清楚展示以下核心概念：

| 概念 | 說明 |
|------|------|
| **前向傳播** | 透過線性方程式計算預測值 |
| **損失函數** | 均方誤差（MSE）衡量預測誤差 |
| **反向傳播** | 自動微分計算梯度 |
| **梯度下降** | 以小批次 SGD 更新參數 |
| **GPU 加速** | 自動偵測並使用 CUDA 裝置 |

---

## 2. 數學原理

### 線性迴歸模型

給定輸入特徵向量 x ∈ R^d，線性迴歸預測輸出為：

```
ŷ = x^T · w + b
```

其中：
- `w ∈ R^d`：權重向量（weight vector）
- `b ∈ R`：偏置（bias）

本例中 d = 2，真實參數為 `w = [2, -3.4]`，`b = 4.2`。

### 損失函數（均方誤差）

```
L(w, b) = ½ · (ŷ - y)²
```

透過最小化此損失，模型學習趨近於真實參數。

### 小批次隨機梯度下降（Mini-batch SGD）

在每個批次計算梯度後，以下式更新參數：

```
w ← w - (η / |B|) · Σ ∇_w L^(i)
b ← b - (η / |B|) · Σ ∇_b L^(i)
```

其中 η 為學習率，|B| 為批次大小。

---

## 3. 程式碼架構

```
linear_regression.py
│
├── [環境設定]    device 選擇（CUDA / CPU）
├── [資料生成]    d2l.synthetic_data → features, labels
├── [函式定義]
│   ├── data_iter()     — 隨機小批次資料迭代器
│   ├── linreg()        — 線性模型前向傳播
│   ├── squared_loss()  — 均方誤差損失
│   └── sgd()           — 小批次梯度下降優化器
├── [參數初始化]  w, b（建立於目標裝置上）
└── [訓練迴圈]    前向 → 損失 → 反向 → 更新 → 評估
```

---

## 4. 逐步說明

### 4.1 環境設定與 GPU 支援

```python
import numpy as np
import random
import torch
from torch.utils import data
from d2l import torch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- **`torch.cuda.is_available()`**：在執行時自動偵測系統是否有可用的 NVIDIA GPU。
- 若有 GPU，`device = 'cuda'`，所有張量計算將在 GPU 上執行；否則退回 CPU。
- 此設計讓程式碼在 CPU 與 GPU 環境下均能無縫運行，無需手動更改。

### 4.2 合成資料集生成

```python
true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = d2l.synthetic_data(true_w, true_b, 100000)
features, labels = features.to(device), labels.to(device)
```

- **`d2l.synthetic_data`**：生成符合 `y = x^T·w + b + ε` 的合成資料，其中 `ε ~ N(0, 0.01)` 為高斯雜訊。
- 共生成 **100,000** 筆資料，每筆有 2 個特徵。
- **`.to(device)`**：將資料搬移到目標裝置（GPU/CPU），確保後續計算在同一裝置上進行。

> **重要**：所有的張量（features、labels、w、b）必須在同一個裝置上，否則 PyTorch 會拋出 RuntimeError。

### 4.3 資料迭代器 `data_iter`

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

這是一個 Python **生成器（generator）**，每次呼叫 `next()` 時回傳一個小批次的資料。

| 步驟 | 說明 |
|------|------|
| `list(range(num_examples))` | 建立索引清單 `[0, 1, 2, ..., N-1]` |
| `random.shuffle(indices)` | 隨機打亂索引，確保每個 epoch 資料順序不同 |
| `yield features[batch_indices]` | 依照批次大小逐批回傳特徵與標籤 |

> **注意**：使用 `yield` 而非 `return`，使函式成為惰性求值的生成器，節省記憶體使用。

### 4.4 線性迴歸模型 `linreg`

```python
def linreg(X, w, b):
    return torch.matmul(X, w) + b
```

- **`torch.matmul(X, w)`**：矩陣乘法，計算 `X · w`。
  - `X` 的形狀：`(batch_size, 2)`
  - `w` 的形狀：`(2,)`
  - 結果形狀：`(batch_size,)`
- 加上偏置 `b` 後，得到預測值 ŷ。

### 4.5 損失函數 `squared_loss`

```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

- 計算每一筆資料的平方損失 `½ · (ŷ - y)²`。
- **`y.reshape(y_hat.shape)`**：確保 `y`（標籤）與 `y_hat`（預測值）的形狀相同，避免廣播（broadcast）錯誤。
- 回傳的是逐元素損失向量，**尚未對批次取平均**，後續在訓練迴圈中呼叫 `.sum()` 進行加總。

> **警告**：括號位置至關重要：必須寫 `(y_hat - y.reshape(...)) ** 2`，而非 `y_hat - y.reshape(...) ** 2`，後者是常見的 bug。

### 4.6 優化器 `sgd`

```python
def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

- **`torch.no_grad()`**：停用此區塊內的梯度追蹤，因為參數更新本身不應被計入計算圖。
- **`param -= lr * param.grad / batch_size`**：梯度下降更新公式。
  - `param.grad` 由 `loss.sum().backward()` 自動填入。
  - 除以 `batch_size` 是對批次損失取平均。
- **`param.grad.zero_()`**：清空梯度，防止梯度在不同批次間累積（PyTorch 預設行為是累積梯度）。

### 4.7 參數初始化

```python
batch_size = 10000
lr = 0.03

w = torch.tensor([0.0, 0.0], requires_grad=True, device=device)
b = torch.zeros(1, requires_grad=True, device=device)
```

- **`w`** 初始化為零向量，**`b`** 初始化為 0。
- **`requires_grad=True`**：告知 PyTorch 需為這些張量追蹤梯度，這是反向傳播的前提。
- **`device=device`**：直接在目標裝置上建立張量，避免先在 CPU 建立再搬移的額外開銷。

### 4.8 訓練迴圈

```python
EPOCH = 40

for epoch in range(EPOCH):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = linreg(X, w, b)          # 1. 前向傳播
        loss = squared_loss(y_hat, y)    # 2. 計算損失
        loss.sum().backward()            # 3. 反向傳播（計算梯度）
        sgd([w, b], lr, batch_size)      # 4. 更新參數

    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

每個 epoch 訓練步驟圖示：

```
┌──────────────────────────────────────────────────────┐
│  for each mini-batch:                                │
│                                                      │
│  [1] 前向傳播  y_hat = X·w + b                       │
│        ↓                                             │
│  [2] 損失計算  loss = ½(ŷ - y)²                      │
│        ↓                                             │
│  [3] 反向傳播  loss.sum().backward()                  │
│      → 自動計算 ∂loss/∂w, ∂loss/∂b                   │
│        ↓                                             │
│  [4] 參數更新  w -= lr * w.grad / batch_size          │
│               b -= lr * b.grad / batch_size          │
└──────────────────────────────────────────────────────┘
```

epoch 結束後，在整個資料集上評估損失並印出，方便追蹤收斂狀況。

---

## 5. 超參數說明

| 超參數 | 目前值 | 說明 | 調整建議 |
|--------|--------|------|----------|
| `batch_size` | `10000` | 每次更新使用的樣本數 | 越大越穩定但記憶體需求高；越小則更新更頻繁但可能震盪 |
| `lr` | `0.03` | 學習率，控制每次參數更新的步伐大小 | 過大會發散，過小會收斂慢；可嘗試 `0.01`~`0.1` |
| `EPOCH` | `40` | 訓練總輪數 | 若損失已收斂可提早停止；若損失仍下降可增加 |
| 資料量 | `100000` | 生成合成資料的樣本數 | 增加資料量有助提升泛化能力 |

---

## 6. GPU 加速原理

```
CPU 模式                          GPU (CUDA) 模式
──────────────────                ───────────────────────────
features → RAM                    features → VRAM (GPU Memory)
labels   → RAM                    labels   → VRAM
w, b     → RAM                    w, b     → VRAM
matmul   → CPU cores              matmul   → CUDA cores (並行)
```

- GPU 擁有數千個小型運算核心，非常擅長**大規模矩陣運算**的並行計算。
- 當 `batch_size` 很大（如 10000）時，一次矩陣乘法計算量可充分利用 GPU 的並行優勢。
- 透過 `.to(device)` 將張量搬移到 GPU 後，所有後續計算都在 GPU 上進行，直到明確搬回 CPU。

---

## 7. 執行結果與收斂

訓練成功後，終端輸出類似：

```
epoch 1,  loss 9.035202
epoch 5,  loss 0.785501
epoch 10, loss 0.037128
epoch 20, loss 0.000133
epoch 30, loss 0.000050
epoch 40, loss 0.000050

true_w: tensor([ 2.0000, -3.4000]), true_b: 4.2
w: tensor([ 2.0000, -3.4000], device='cuda:0'), b: tensor([4.2000], device='cuda:0')
```

- 約在第 **25~30 個 epoch** 後損失趨於穩定（約 `5e-5`，由資料雜訊 ε 決定下限）。
- 學習到的 `w` 與 `b` 非常接近真實值，驗證實作正確。
- `device='cuda:0'` 確認計算確實在 GPU 上完成。

---

## 8. 常見問題

**Q1：為什麼 loss 不會收斂到 0？**

因為合成資料加入了高斯雜訊 `ε ~ N(0, 0.01)`，即使找到完美參數，雜訊本身也會造成一個固定的最小損失值。

---

**Q2：為什麼要在 `sgd` 中使用 `torch.no_grad()`？**

`param -= ...` 這個原地操作（in-place operation）是對參數的直接修改，不應被 PyTorch 的自動微分計算圖所追蹤，否則會造成計算圖污染或記憶體浪費。

---

**Q3：`batch_size` 設太大有什麼問題？**

若 `batch_size` 等於整個資料集大小，就變成了**批次梯度下降（Batch GD）**，每個 epoch 只更新一次，雖然 GPU 並行效率高，但缺乏隨機性，可能陷入局部最優。

---

**Q4：如何強制使用 CPU 測試？**

```python
device = torch.device('cpu')
```

---

**Q5：`with torch.no_grad():` 的用途是什麼？**

這是 PyTorch 的**上下文管理器（context manager）**，用來**暫時停用自動微分（autograd）的梯度追蹤**。

PyTorch 預設會為每個張量運算建立**計算圖（computation graph）**，以便事後呼叫 `.backward()` 計算梯度。這個計算圖雖是反向傳播的基礎，但它**佔用記憶體且有效能開銷**。使用 `torch.no_grad()` 可在不需要梯度的場景中關閉這個機制。

程式碼中共出現兩處：

**第一處：`sgd` 優化器中（參數更新）**

```python
def sgd(params, lr, batch_size):
    with torch.no_grad():       # ← 停用梯度追蹤
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

參數更新（`w -= ...`）是直接修改數值的操作，不是模型的一部分，不需要被計入計算圖。若不加這行，PyTorch 會對「更新步驟」也建立計算圖，造成記憶體浪費，甚至引發 in-place operation 錯誤。

**第二處：訓練迴圈中（評估損失）**

```python
with torch.no_grad():           # ← 停用梯度追蹤
    train_l = squared_loss(linreg(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

此處只是計算損失來**觀察訓練進度**，完全不需要反向傳播，因此不需要建立計算圖。

**使用時機總結：**

| 場景 | 需要梯度追蹤？ | 理由 |
|------|---------------|------|
| 前向傳播 `linreg(X, w, b)` | ✅ 需要 | 要能 `.backward()` 計算梯度 |
| 損失計算 `squared_loss(...)` | ✅ 需要 | 要能 `.backward()` 計算梯度 |
| 參數更新 `param -= ...` | ❌ 不需要 | 只是修改數值，非模型運算 |
| 評估損失（印出 log） | ❌ 不需要 | 只是觀察，不做反向傳播 |

> **口訣**：凡是「不需要呼叫 `.backward()` 的計算」，都應包在 `torch.no_grad()` 裡，可大幅減少記憶體使用，在模型推論（inference）階段尤為重要。

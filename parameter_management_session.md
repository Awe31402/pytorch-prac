# 深度學習學習日誌：參數管理
**日期：** 2026-04-14
**專案路徑：** `/home/awe/pytorch`

---

## 1. 參數管理 (Parameter Management) 學習重點

根據《Dive into Deep Learning》第 5.2 節，參數管理包含以下三大支柱：

### A. 參數訪問 (Access)
*   **`state_dict()`**：獲取包含參數名稱與張量的字典。
*   **`named_parameters()`**：遞歸遍歷所有層，取得名稱與參數對象。
*   **`.data` 屬性**：直接訪問參數的數值張量（不包含梯度資訊）。

### B. 參數初始化 (Initialization)
*   **`net.apply(fn)`**：這是最重要的工具，能將初始化函數遞歸應用到網路的每一個子模組。
*   **內建方法**：PyTorch 提供 `nn.init.normal_`, `nn.init.xavier_uniform_`, `nn.init.kaiming_uniform_` 等。

### C. 參數綁定 (Tied Parameters)
*   多個層共享同一個 `nn.Parameter` 對象。
*   **特性**：修改其中一個，其他層會同步改變；反向傳播時，綁定層的梯度會**累加**。

---

## 2. 深度討論：Xavier vs. Kaiming 初始化

### Xavier (Glorot) 初始化
*   **公式**：$\text{std} = \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$
*   **原理**：保持輸入與輸出的變異數一致。
*   **適用場景**：S 型激活函數（Sigmoid, Tanh）。

### Kaiming (He) 初始化
*   **公式**：$\text{std} = \sqrt{\frac{2}{\text{fan\_in}}}$
*   **原理**：補償 ReLU 丟棄一半負數訊號導致的變異數減半問題。
*   **適用場景**：**ReLU**, Leaky ReLU 等現代深度網路首選。

---

## 3. 今日實作腳本：`parameter_init_comparison.py`

```python
import torch
from torch import nn

def get_net():
    return nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

def init_weights_xavier(m):
    if type(m) == nn.Linear:
        # Xavier 均勻分佈初始化
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def main():
    net = get_net()
    print("--- 開始執行 Xavier 初始化 ---")
    net.apply(init_weights_xavier)
    print("初始化完成。")

if __name__ == "__main__":
    main()
```

---
**日誌總結：** 今天深入理解了參數管理的機制，掌握了參數初始化的數學背景與 PyTorch 實作技巧，並能針對不同的激活函數選擇合適的初始化策略。

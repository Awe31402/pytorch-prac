# 深度學習計算：層和塊 (Layers and Blocks) 解析

在《動手學深度學習》(D2L) 第五章中，詳細探討了深度學習計算的核心建構單元——**塊（Block）**。塊可以是單一個神經網路層、由多個層組成的組件，或是整個模型本身。

本文檔結合了 [dl_nn_blocks.py](file:///home/awe/pytorch/dl_nn_blocks.py) 中的實作，來解析 PyTorch 中 `nn.Module` 的底層行為，以及自定義層與順序塊的運作原理。

---

## 1. 使用 `nn.Module` 自定義神經網路塊 (MLP)

從程式設計的角度來看，塊是由**類別 (Class)** 來表示的。在 PyTorch 中，任何神經網路的層或塊都必須繼承自 `torch.nn.Module`。自定義一個塊，我們主要需要完成兩件事：
1. **初始化參數和子模塊 (`__init__`)**：設定網路層的輸入、輸出維度等結構。
2. **定義前向傳播邏輯 (`forward`)**：決定資料（張量）如何從網路的輸入端，一步步轉換至輸出端。

### 程式碼對照解析：多層感知機 (MLP)

```python
class MLP(nn.Module):
    def __init__(self):
        # 必須呼叫父類的建構式，這涉及 PyTorch 底層參數註冊和初始化的準備
        super().__init__()
        # 聲明隱藏層及輸出層，兩者皆為 Linear 子類，也是一個 Module
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        # 輸入 X 通過隱藏層，並以 F.relu(ReLU 函式的寫法) 增添非線性特徵
        # 最後結果再傳至輸出層 (self.out)
        return self.out(F.relu(self.hidden(X)))
```

> [!NOTE]
> 在這個 `MLP` 中，當呼叫 `net(X)` 時，實際上這是在呼叫 `net.__call__(X)` 並在內部觸發執行了我們自定義的 `forward(X)`。

---

## 2. 深入剖析連續串接：順序塊 (`MySequential`)

為了解 `nn.Sequential` 究竟如何在幕後自動為我們連接好所有的層，在第五章引入了 `MySequential` 作為原理探討。

### 如何將未知的子層註冊進網路架構中？

PyTorch 要求你必須將子模塊明確掛載在 `self` 屬性下。在不知道有幾個層的情況下，順序塊的關鍵在於如何妥善收集這些類別變數。在 `nn.Module` 中，預設維護了一個特殊的底層字典：`self._modules`。

### 程式碼對照解析：自定義順序塊

```python
class MySequential(nn.Module):
    # MySequential 的設計目的是把作為參數傳入的其他模塊依序串聯起來
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # args 中傳入的是 nn.Module 類別的實體
            # 將其透過 idx 轉為字串鍵值，放入 `_modules` 有序字典中
            self._modules[str(idx)] = module

    def forward(self, X):
        # 因為 OrderedDict (有序字典) 會記住元素加入的順序
        # 直接使用 values() 遍歷所有層，依序將上層的輸出做為下層的輸入
        for block in self._modules.values():
            X = block(X)
        return X
```

### 為什麼不直接用 Python 原生的串列 (List)？

> [!IMPORTANT]
> `_modules` 字典具有特殊的意義！
> 系統（包含自動求導、保存/加載模型的功能）在尋找模型參數（如 Weights 和 Bias）時，**只會檢查被註冊到 `_modules` 或 `_parameters` 中的內容**。如果您只使用 Python 的普通 `list` 儲存子模塊，在做 `net.parameters()` 取權重進行優化算法時將找不到任何變數。

---

## 3. 測試與執行結果探討

在 [dl_nn_blocks.py:L36-62](file:///home/awe/pytorch/dl_nn_blocks.py#L36-L62) 中，我們對這兩種自定義寫法進行了測試：

1. **直接實作 MLP**: 對於複雜的資料流，比如引入條件分歧 `if...else` 或者常數計算，使用像 `MLP` 這種完整自訂 `forward` 函式的方法會給予最高的彈性。
2. **利用 `MySequential` (或 `nn.Sequential`)**: 適用於資料處理具有絕對線性流程的模型。它可以避免手寫繁瑣的 `forward` 函式。

結果如下：
```diff
輸出形狀: torch.Size([2, 10])
...
MySequential 輸出形狀: torch.Size([2, 10])
```

另外，透過直接 `print(net)` 列印模型實體時，可以發現：
```text
MLP(
  (hidden): Linear(in_features=20, out_features=256, bias=True)
  (out): Linear(in_features=256, out_features=10, bias=True)
)
```
與 
```text
MySequential(
  (0): Linear(in_features=20, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
```
這也是 `self._modules` 自動整理好的結果：前者把層對應在命名好的物件變數 (`hidden`, `out`) 上，而後者則是依照串聯順序 (`0`, `1`, `2`) 作為變數名稱註冊。

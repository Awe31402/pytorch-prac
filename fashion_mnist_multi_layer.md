# Fashion MNIST：多層感知機 (MLP) 實作說明

本文件主要說明 `fashion_mnist_multi_layer.py` 的實作細節，並對照原本的 `fashion_mnist.py` (Softmax 迴歸) 來解析兩者在神經網路設計與程式碼層面上的差異。

## 1. 網路架構 (Network Architecture)

*   **`fashion_mnist.py` (Softmax Regression)**：
    單層線性模型，輸入的圖片像素 (784 維) 經過一次線性變換後，直接映射到 10 個類別輸出。沒有隱藏層，無法學習特徵之間的非線性複雜關係。
*   **`fashion_mnist_multi_layer.py` (MLP)**：
    加入了一層擁有 256 個神經元的**隱藏層 (Hidden Layer)**。網路架構由原本的 `784 -> 10` 變為 `784 -> 256 -> 10`。

## 2. 參數初始化 (Parameter Initialization)

*   **`fashion_mnist.py`**：
    只需維護一組權重與偏差值：
    ```python
    W = torch.normal(0, 0.01, size=(num_input, num_output), ...)
    b = torch.zeros(num_output, ...)
    ```
*   **`fashion_mnist_multi_layer.py`**：
    為了構建多層架構，需要兩組權重矩陣與偏差向量：
    *   隱藏層參數：`W1` (784x256), `b1` (256)
    *   輸出層參數：`W2` (256x10), `b2` (10)

## 3. 非線性激活函數 (Activation Function)

*   **`fashion_mnist.py`**：完全維持線性運算，僅在最後計算 Loss 時做 Softmax 分佈。
*   **`fashion_mnist_multi_layer.py`**：
    在隱藏層輸出的後面加入了 **ReLU (Rectified Linear Unit)** 函數，以引入非線性。若沒有 ReLU，多層矩陣相乘等價於單層線性回歸，無法發揮多層網路的優勢。
    ```python
    def relu(X):
        a = torch.zeros_like(X)
        return torch.max(X, a)
        
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1) # 隱藏層 + ReLU 激活
        return (H @ W2 + b2)  # 輸出層
    ```

## 4. 損失函數 (Loss Function) 的計算方式

*   **`fashion_mnist.py`**：
    手動實作了 `softmax` 函數以及 `cross_entropy` 計算。在反向傳播遇到極端數值（如浮點數溢位）時容易產生不穩定。
*   **`fashion_mnist_multi_layer.py`**：
    移除了手動版函式，直接調用 PyTorch 內建的高階 API `nn.CrossEntropyLoss(reduction='none')`。這個函式在內部將 Softmax 與 Cross Entropy 合併計算，從而確保數值穩定性（Log-Sum-Exp 技巧）。因此 `net(X)` 最後不需要回傳 Softmax 的結果，只要直接回傳 `logits` 即可。

## 5. 梯度更新 (Optimizer Updates)

*   **`fashion_mnist.py`**：
    在每一個批次 (Batch) 訓練後，使用自定義的 `SGD` 函數更新 `[W, b]`。
*   **`fashion_mnist_multi_layer.py`**：
    神經網路層數增加，需要在 SGD 中將所有的 trainable parameters 傳入更新：
    ```python
    SGD([W1, b1, W2, b2], lr, batch_size)
    ```

---

## 執行結果比較與結語

透過增加一層 256 個神經元的隱藏層與非線性的 ReLU 激活函數，多層感知機 (MLP) 的模型容量（Capacity）變得更大。經過相同的 20 世代 (Epochs) 訓練，MLP 模型通常可以學習到比單層 Softmax 迴歸更加抽象的映射規則。從終端機輸出來看，MLP 模型通常會有更低的 Loss，並在訓練集、測試集上擁有更優秀的準確率表現。

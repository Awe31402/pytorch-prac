import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    # 用模型參數聲明層。這裡我們聲明兩個全連接層。
    def __init__(self):
        # 調用 MLP 的父類 nn.Module 的構造函數來執行必要的初始化。
        # 這樣在類實例化時也可以指定其他函數參數，例如模型參數。
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隱藏層
        self.out = nn.Linear(256, 10)     # 輸出層

    # 定義模型的前向傳播，即如何根據輸入 X 返回所需的模型輸出
    def forward(self, X):
        # 這裡我們使用 ReLU 的函數版本，其在 torch.nn.functional 模塊中定義。
        # 將輸入通過隱藏層，經 ReLU 激活後，再通過輸出層。
        return self.out(F.relu(self.hidden(X)))

class MySequential(nn.Module):
    # MySequential 旨在將其他模塊串聯起來
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 這裡的 module 是 nn.Module 子類的實例
            # 我們將其保存在 nn.Module 類的成員變數 _modules 中
            # _modules 的類型是 OrderedDict，系統會自動搜尋它來初始化參數
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict 保證了按照成員被添加的順序遍歷它們
        for block in self._modules.values():
            X = block(X)
        return X

if __name__ == '__main__':
    # 實例化多層感知機模型
    net = MLP()
    
    # 創建一個形狀為 (2, 20) 的隨機張量作為輸入，代表 2 個樣本，每個樣本 20 個特徵
    X = torch.rand(2, 20)
    
    # 執行前向傳播以生成輸出
    output = net(X)
    
    print("輸入形狀:", X.shape)
    print("輸出形狀:", output.shape)
    print("運算結果:\n", output)
    print(net)

    print("\n" + "="*30)
    print("測試 MySequential")
    # 使用我們自定義的 MySequential 類重新實現多層感知機
    net_seq = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    
    # 將相同的輸入 X 傳遞給它
    output_seq = net_seq(X)
    print("MySequential 輸出形狀:", output_seq.shape)
    print("MySequential 運算結果:\n", output_seq)

    # 打印模型結構
    print(net_seq)
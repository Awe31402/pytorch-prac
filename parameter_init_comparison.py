import torch
from torch import nn

# 1. 定義網路架構
def get_net():
    return nn.Sequential(
        nn.Linear(20, 256), 
        nn.ReLU(), 
        nn.Linear(256, 10)
    )

# 2. 定義 Xavier 初始化函數
def init_weights_xavier(m):
    if type(m) == nn.Linear:
        print(f"正在初始化層: {m}")
        # 使用 Xavier 均勻分佈初始化權重
        nn.init.xavier_uniform_(m.weight)
        # 偏置通常初始化為 0
        nn.init.zeros_(m.bias)

def main():
    # 建立模型
    net = get_net()
    
    print("--- 初始化前的權重 (前 5 個數值) ---")
    print(net[0].weight.data[0][:5])
    
    # 3. 套用初始化
    print("\n--- 開始執行自定義初始化 ---")
    net.apply(init_weights_xavier)
    
    print("\n--- 初始化後的權重 (前 5 個數值) ---")
    print(net[0].weight.data[0][:5])
    
    # 4. 驗證參數綁定 (額外練習)
    print("\n--- 參數管理：檢查權重梯度狀態 ---")
    print(f"權重是否需要梯度: {net[0].weight.requires_grad}")

if __name__ == "__main__":
    main()

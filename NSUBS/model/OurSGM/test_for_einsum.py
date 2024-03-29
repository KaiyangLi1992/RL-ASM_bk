import torch
import numpy as np
import random

# 设置全局随机种子
seed = 42

# Python 的内置随机库
random.seed(seed)

# Numpy 随机种子
np.random.seed(seed)

# PyTorch 随机种子
torch.manual_seed(seed)


A = torch.rand(2, 5)
B = torch.rand(3, 5)
M = torch.rand(5, 5, 32)

A_= A[1, :].unsqueeze(0)
B_= B[2,:].unsqueeze(0)

sim_latent = torch.einsum('ik,klj,hl->ihj', A_, M,B_)
print(sim_latent) 
correspondence = torch.tensor([0, 0, 1])



# 根据对应关系向量进行相应的操作
A_corresponding = A[correspondence]
# _
# 使用广播机制对 A 和 B 进行相乘，并应用 M
A = torch.einsum('ik,klj->ilj', A_corresponding , M)
print(A.shape)  # 输出结果的形状

B = B.unsqueeze(-1)

# 进行广播乘法
result = A * B  # 结果的尺寸为 3x5x32

# 沿着第二个维度求和，得到最终结果
result_final = result.sum(dim=1)  # 最终结果的尺寸为 3x32

print(result_final.shape)  # 输出结果的形状
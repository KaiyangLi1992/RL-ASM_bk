import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, Batch
import numpy as np
from torch_scatter import scatter_mean,scatter_sum,scatter
import sys
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/") 
from NSUBS.model.OurSGM.utils_nn import MLP, get_MLP_args, NormalizeAttention
def normalize_logits(x, batch,epsilon=0.001):
    mean_features = scatter_mean(x, batch, dim=0)
    expanded_mean_features = mean_features[batch]
    centered = x - expanded_mean_features
    sum_features = scatter_sum(centered ** 2, batch, dim=0)
    expanded_sum_features = sum_features[batch]

    src = torch.ones_like(batch, dtype=torch.float)
    counts = scatter(src, batch, dim=0, reduce="sum")
    counts =  counts[batch].unsqueeze(1)

    std = torch.sqrt(expanded_sum_features/counts)+epsilon
    return centered / std
class NormalizeAttention(torch.nn.Module):
    def __init__(self):
        super(NormalizeAttention, self).__init__()
        self.gain = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, inputs,batch):
        logits = inputs.view(-1,1)
        return self.gain * normalize_logits(logits,batch) + self.bias
torch_geometric.seed_everything(0)
# 假设的模型部分，包含MLP层
class SimpleGraphModel(nn.Module):
    def __init__(self):
        super(SimpleGraphModel, self).__init__()
        self.mlp_att = nn.Linear(5, 1) # 假设每个节点特征是5维的，输出1维注意力分数
        self.mlp_val = nn.Linear(5, 5) # 节点值向量，假设输入输出都是5维
        self.norm = NormalizeAttention()
        self.mlp_final = nn.Linear(5, 5) # 最终MLP，假设输入输出都是5维

    # def forward(self, pyg_data_q):
    #     Q_li, V_li = [], []
    #     unique_batch_indices = pyg_data_q.batch.unique(sorted=True)
    #     for graph_index in unique_batch_indices:
    #         graph_node_indices = (pyg_data_q.batch == graph_index).nonzero(as_tuple=True)[0]
    #         Xq = pyg_data_q.x[graph_node_indices, :]
    #         Q = torch.sum(self.norm(self.mlp_att(Xq)).view(-1, 1) * self.mlp_val(Xq), dim=0) / Xq.shape[0]
    #         V = F.leaky_relu(self.mlp_final(Q))
    #         Q_li.append(Q)
    #         V_li.append(V)
    #     return torch.stack(Q_li), torch.stack(V_li)
    
    def forward(self, pyg_data_q):
    # 应用MLP并计算注意力得分
        att_scores = self.mlp_att(pyg_data_q.x)
        att_scores = self.norm(att_scores,pyg_data_q.batch)
        
        # 应用值MLP
        values = self.mlp_val(pyg_data_q.x)
        
        # 计算加权值
        weighted_values = att_scores * values
        
        # 聚合每个图的节点特征，生成图级特征
        Q = scatter_mean(weighted_values, pyg_data_q.batch, dim=0)
        
        # 应用最终MLP
        V = F.leaky_relu(self.mlp_final(Q))

        return Q, V

# 测试函数
def test_simple_graph_model():
    # 创建假设的节点特征和batch索引
    x = torch.rand((10, 5)) # 假设有10个节点，每个节点5维特征
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 假设这10个节点属于2个图

    # 创建PyG Data对象
    data_list = [Data(x=x[batch==i], batch=batch[batch==i]) for i in batch.unique()]
    pyg_data_q = Batch.from_data_list(data_list)

    model = SimpleGraphModel()
    Q_li, V_li = model(pyg_data_q)

    assert Q_li.shape[0] == 2 and Q_li.shape[1] == 5, "输出Q的形状应该是(2, 5)"
    assert V_li.shape[0] == 2 and V_li.shape[1] == 5, "输出V的形状应该是(2, 5)"
    print("测试通过!")

test_simple_graph_model()


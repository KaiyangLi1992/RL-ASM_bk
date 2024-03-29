import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch_geometric
torch_geometric.seed_everything(0)

def transform_u(u, batch):
    batch = batch.tolist()
    start_indices = {}
    current_batch_id = None
    for i, batch_id in enumerate(batch):
        if batch_id != current_batch_id:
            start_indices[batch_id] = i
            current_batch_id = batch_id
    u_new = [start_indices[i] + index for i, index in enumerate(u)]
    return u_new

def transform_v_li(v_li, batch):
    batch = batch.tolist()
    start_indices = {}
    current_batch_id = None
    for i, batch_id in enumerate(batch):
        if batch_id != current_batch_id:
            start_indices[batch_id] = i
            current_batch_id = batch_id

    v_li_new = [[start_indices[i] + index for index in v_li_sp] for i, v_li_sp in enumerate(v_li)]
    flattened_v_li_new = [item for sublist in v_li_new for item in sublist]
    return flattened_v_li_new

def correspondence_li(two_dimensional_list):
    index_list = [i for i, sublist in enumerate(two_dimensional_list) for _ in sublist]
    return index_list

def split_tensor(t, corres):
    unique_indices = torch.unique(torch.tensor(corres))

    # 根据索引列表分割张量并存入新列表
    split_tensors = [t[torch.tensor(corres) == index] for index in unique_indices]
    return  split_tensors





class BilinearDecoder(torch.nn.Module):
    def __init__(self, mlp_in, mlp_out, d_in, d_out):
        super(BilinearDecoder, self).__init__()
        self.encoder = mlp_in
        self.bilinear_mat = torch.nn.Parameter(torch.randn(d_in, d_in, d_out) + torch.eye(d_in).view(d_in, d_in, 1))
        self.decoder = mlp_out

    def forward(self, pyg_data_q, pyg_data_t, u, v_li, g_emb):
        v_corres = correspondence_li(v_li)  
        u = transform_u(u, pyg_data_q.batch)
        v_li = transform_v_li(v_li, pyg_data_t.batch)
        
        g_emb_sp = g_emb[v_corres]
        

        u_emb = pyg_data_q.x[u,:]
        v_emb = pyg_data_t.x[v_li,:]
        Xsgq_latent, Xsgt_latent = self.encoder(u_emb), self.encoder(v_emb)

        Xsgq_latent_corres = Xsgq_latent[v_corres]
        # _
        # 使用广播机制对 A 和 B 进行相乘，并应用 M
        A = torch.einsum('ik,klj->ilj', Xsgq_latent_corres , self.bilinear_mat)
        B = Xsgt_latent.unsqueeze(-1)

        result = A * B  # 结果的尺寸为 3x5x32
        sim_latent = result.sum(dim=1)  # 最终结果的尺寸为 3x32

        sim = self.decoder(torch.cat((sim_latent,g_emb_sp), dim=1))
        split_tensors = split_tensor(sim.view(-1), v_corres)
        for i,out in enumerate(split_tensors):
            policy_truth = torch.zeros(out.shape)
            _
               
        _

                


        # sim_li, sim_latent_li = [],[]
        # unique_batch_indices = pyg_data_q.batch.unique(sorted=True)
        # for graph_index in unique_batch_indices:
        #     g_emb_sp = g_emb[graph_index]

        #     u_sp = u[graph_index]
        #     v_li_sp = v_li[graph_index]

        #     Xq = pyg_data_q.x                
        #     graph_node_indices_q = (pyg_data_q.batch == graph_index).nonzero(as_tuple=True)[0]
        #     Xq  = Xq[graph_node_indices_q,:] 

        #     Xt = pyg_data_t.x
        #     graph_node_indices_t = (pyg_data_t.batch == graph_index).nonzero(as_tuple=True)[0]
        #     Xt  = Xt[graph_node_indices_t,:] 

        #     Xsgq_latent, Xsgt_latent = self.encoder(Xq[[u_sp]]), self.encoder(Xt[v_li_sp])
        #     sim_latent = torch.einsum('ik,klj,hl->ihj', Xsgq_latent.unsqueeze(0), self.bilinear_mat, Xsgt_latent)
        #     sim = self.decoder(torch.cat((sim_latent, g_emb_sp.view(1,1,-1).repeat(*sim_latent.shape[:-1], 1)), dim=2))

        #     sim_li.append(sim)
        #     sim_latent_li.append(sim_latent.squeeze(dim=0))

        # return sim_li, sim_latent_li
    



    # 假设的MLP编码器和解码器
class DummyMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DummyMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# BilinearDecoder类（已提供）

# 创建模拟的图结构数据
num_nodes = 10
num_features = 5
num_graphs = 2
d_in, d_out = 5, 3

# 为每个图创建节点特征和batch指示器
x = torch.randn(num_nodes * num_graphs, num_features)
batch = torch.tensor([i // num_nodes for i in range(num_nodes * num_graphs)])

# 创建图数据对象
pyg_data_q = Data(x=x, batch=batch)
pyg_data_t = Data(x=x, batch=batch)

# 创建节点嵌入和全局嵌入
u = [0, 5] # 假设每个图选一个节点作为u
v_li = [[1, 2, 3], [6, 7, 8]] # 每个图有多个v节点
g_emb = torch.randn(num_graphs, d_in) # 每个图的全局嵌入

# 实例化BilinearDecoder
encoder = DummyMLP(num_features, d_in)
decoder = DummyMLP(d_in + d_out, 1) # 假设输出是一个标量值
model = BilinearDecoder(encoder, decoder, d_in, d_out)

# 调用forward方法
sim_li, sim_latent_li = model.forward(pyg_data_q, pyg_data_t, u, v_li, g_emb)

# 打印结果
for sim in sim_li:
    print(sim.size())

for sim_latent in sim_latent_li:
    print(sim_latent.size())
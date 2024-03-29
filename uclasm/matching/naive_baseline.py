import  pickle
import networkx as nx
from networkx.algorithms import isomorphism

with open('./data/SYNTHETIC/for_AEDNET/SYNTHETIC_pairs_noiseratio_5.0.pkl','rb') as f:
    pairs = pickle.load(f)
with open('./data/SYNTHETIC/for_AEDNET/SYNTHETIC_graph_dict_noiseratio_5.0.pkl','rb') as f:
    graph_dict = pickle.load(f)
# with open('./data/AIDS/AIDS_dataset_sparse_noiseratio_n_6_10_num_01_31_matching.pkl','rb') as f:
#     matchs = pickle.load(f)
    

def check_sets(sets_list):
    # 找到最大集合的大小
    max_size = max(len(s) for s in sets_list)
    
    # 计数最大集合的数量
    max_sets_count = sum(1 for s in sets_list if len(s) == max_size)
    
    # 如果有超过一个最大集合，直接返回False
    if max_sets_count > 1:
        return False
    
    # 检查除了最大集合外的其他集合的大小是否都为1
    for s in sets_list:
        if len(s) != max_size and len(s) != 1:
            return False
            
    return True

# 示例
sets_list = [{1, 2, 3}, {4}, {5}, {6}]
result = check_sets(sets_list)
print(result)  # 应该打印True或False，根据实际情况

    
def remain_largest_component(G):
    connected_components = list(nx.connected_components(G))
    if check_sets(connected_components):
        # 保留最大的连通分量
        largest_cc = max(connected_components, key=len)

        # 创建一个新图，只包含最大的连通分量中的节点和边
        G_sub = G.subgraph(largest_cc).copy()
        return G_sub,True
    else:
        return None,False

    

def calculate_distance(g1,g2):
    edges_to_remove = list(g1.edges())
        # 遍历列表并删除每条边
    costs = []
    for edge in edges_to_remove:
        g1_ = nx.Graph(g1)
        g1_.remove_edge(*edge)  # 使用*edge来解包边的节点
        g1_ = remain_largest_component(g1_)
        g1_,flag = remain_largest_component(g1_)
        if not flag:
            continue
        matcher = isomorphism.GraphMatcher(g2, g1_,node_match=node_match)
        if matcher.subgraph_is_isomorphic():
            return g1.number_of_nodes()+g1.number_of_edges()-g1_.number_of_edges()-g1.number_of_nodes()
   

    for i in range(len(edges_to_remove)):
        for j in range(i+1,len(edges_to_remove)):
            edge1 = edges_to_remove[i]
            edge2 = edges_to_remove[j]
            g1_ = nx.Graph(g1)
            g1_.remove_edge(*edge1)  # 使用*edge来解包边的节点
            g1_.remove_edge(*edge2) 
            g1_,flag = remain_largest_component(g1_)
            if not flag:
                continue
            matcher = isomorphism.GraphMatcher(g2, g1_,node_match=node_match)
            if matcher.subgraph_is_isomorphic():
                return g1.number_of_nodes()+g1.number_of_edges()-g1_.number_of_edges()-g1.number_of_nodes()
    
    if costs:
        return min(costs)
    for i in range(len(edges_to_remove)):
        for j in range(i+1,len(edges_to_remove)):
            for k in range(j+1,len(edges_to_remove)):
                edge1 = edges_to_remove[i]
                edge2 = edges_to_remove[j]
                edge3 = edges_to_remove[k]
                g1_ = nx.Graph(g1)
                g1_.remove_edge(*edge1)  # 使用*edge来解包边的节点
                g1_.remove_edge(*edge2)
                g1_.remove_edge(*edge3)  # 使用*edge来解包边的节点
             
                g1_ = remain_largest_component(g1_)
                matcher = isomorphism.GraphMatcher(g2, g1_,node_match=node_match)
                if matcher.subgraph_is_isomorphic():
                    costs.append(g1.number_of_nodes()+g1.number_of_edges()-g1_.number_of_edges()-g1.number_of_nodes())
        if costs:
            return min(costs)
        

        for i in range(len(edges_to_remove)):
            for j in range(i+1,len(edges_to_remove)):
                for k in range(j+1,len(edges_to_remove)):
                    for x in range(k+1,len(edges_to_remove)):
                        edge1 = edges_to_remove[i]
                        edge2 = edges_to_remove[j]
                        edge3 = edges_to_remove[k]
                        edge4 = edges_to_remove[x]
                        g1_ = nx.Graph(g1)
                        g1_.remove_edge(*edge1)  # 使用*edge来解包边的节点
                        g1_.remove_edge(*edge2)
                        g1_.remove_edge(*edge3)  # 使用*edge来解包边的节点
                        g1_.remove_edge(*edge4)
                        g1_ = remain_largest_component(g1_)
                        matcher = isomorphism.GraphMatcher(g2, g1_,node_match=node_match)
                        if matcher.subgraph_is_isomorphic():
                            costs.append(g1.number_of_nodes()+g1.number_of_edges()-g1_.number_of_edges()-g1.number_of_nodes())
        if costs:
            return min(costs)
        else:
            return -1
        


        

def node_match(n1, n2):
    # n1是图B中的节点数据，n2是图A中的节点数据
    return n1['type'] == n2['type']


costs = []
for pair in pairs[:100]:
    g1 = graph_dict[pair[0]]
    g2 = graph_dict[pair[1]]
    cost = calculate_distance(g1,g2)
    costs.append(cost)
    print(cost)

    
    
    

costs










        

# import networkx as nx
# from networkx.algorithms import isomorphism

# # 创建主图
# G = nx.Graph()
# # 在主图中添加边
# G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])

# # 创建一个可能的子图
# H = nx.Graph()
# # 在子图中添加边
# H.add_edges_from([(11, 12), (12, 13), (13, 11)])

# # 初始化图匹配器，这里使用的是VF2算法
# matcher = isomorphism.GraphMatcher(G, H)

# # 检查主图G是否包含与H同构的子图
# if matcher.subgraph_is_isomorphic():
#     print("G contains a subgraph that is isomorphic to H")

#     # 获取同构的子图映射
#     subgraph_mappings = list(matcher.subgraph_isomorphisms_iter())
#     print("Isomorphic subgraph mappings:", subgraph_mappings)
# else:
#     print("G does not contain a subgraph that is isomorphic to H")




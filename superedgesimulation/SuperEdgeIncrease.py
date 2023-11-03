import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ===============================第一层网络初始化=============================
# 指定节点数和边数
num_nodes1 = 10
# 创建一个空图
G1 = nx.Graph()

# 使用Barabási-Albert模型生成无标度网络
G1 = nx.barabasi_albert_graph(num_nodes1, 3)
adj_matrix1 = nx.to_numpy_matrix(G1)
print(adj_matrix1)
print(np.sum(adj_matrix1))

# 绘制网络
pos1 = nx.spring_layout(G1)
nx.draw(G1, with_labels=True, node_size=500, node_color='skyblue')
plt.title("Scale-Free Network with Specified Nodes and Edges")
plt.show()


# ===============================第二层网络初始化=============================
# 指定节点数和边数
num_nodes2 = 20
# 创建一个空图
G2 = nx.Graph()

# 使用Barabási-Albert模型生成无标度网络
G2 = nx.barabasi_albert_graph(num_nodes2, 3)
adj_matrix2 = nx.to_numpy_matrix(G2)
print(adj_matrix2)
print(np.sum(adj_matrix2))

# 绘制网络
pos2 = nx.spring_layout(G2)
nx.draw(G2, with_labels=True, node_size=500, node_color='skyblue')
plt.title("Scale-Free Network with Specified Nodes and Edges")
plt.show()

# ===============================第三层网络初始化=============================
# 指定节点数和边数
num_nodes3 = 30
# 创建一个空图
G3 = nx.Graph()

# 使用Barabási-Albert模型生成无标度网络
G3 = nx.barabasi_albert_graph(num_nodes3, 3)
adj_matrix3 = nx.to_numpy_matrix(G3)
print(adj_matrix3)
print(np.sum(adj_matrix3))

# 绘制网络
pos3 = nx.spring_layout(G3)
nx.draw(G3, with_labels=True, node_size=500, node_color='skyblue')
plt.title("Scale-Free Network with Specified Nodes and Edges")
plt.show()


# ===============================第四层网络初始化=============================
# 指定节点数和边数
num_nodes4 = 40
# 创建一个空图
G4 = nx.Graph()

# 使用Barabási-Albert模型生成无标度网络
G4 = nx.barabasi_albert_graph(num_nodes4, 3)
adj_matrix4 = nx.to_numpy_matrix(G4)
print(adj_matrix4)
print(np.sum(adj_matrix4))

# 绘制网络
pos4 = nx.spring_layout(G4)
nx.draw(G4, with_labels=True, node_size=500, node_color='skyblue')
plt.title("Scale-Free Network with Specified Nodes and Edges")
plt.show()


# 邻接矩阵序列3，边数从81升到300
# 邻接矩阵序列1，边数从21升到90
# 邻接矩阵序列2，边数从51升到150
# 邻接矩阵序列4，边数从111升到350


np.save("net_3.npy", adj_matrix3)
np.save("net_1.npy", adj_matrix1)
np.save("net_4.npy", adj_matrix4)
np.save("net_2.npy", adj_matrix2)
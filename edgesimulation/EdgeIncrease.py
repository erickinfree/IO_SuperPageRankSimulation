import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# 生成指定边数的无零行零列的对称矩阵
def generate_random_symmetric_binary_matrix_givensize(matrix, num_ones):
    symmetrical_matrix = matrix
    # 计算当前矩阵中 1 的数目
    current_num_ones = np.sum(symmetrical_matrix)
    print(current_num_ones)
    while np.sum(symmetrical_matrix) < num_ones:
        row, col = np.random.randint(len(symmetrical_matrix)), np.random.randint(len(symmetrical_matrix))
        # 如果当前位置的元素是 0，则设置为 1
        if symmetrical_matrix[row, col] == 0 and row != col:
            symmetrical_matrix[row, col] = 1
            symmetrical_matrix[col, row] = 1
        print(symmetrical_matrix)

    while np.sum(symmetrical_matrix) > num_ones:
        row, col = np.random.randint(len(symmetrical_matrix)), np.random.randint(len(symmetrical_matrix))

        # 如果当前位置的元素是 0，则设置为 1
        if symmetrical_matrix[row, col] == 1 and row != col:
            symmetrical_matrix[row, col] = 0
            symmetrical_matrix[col, row] = 0

        if not np.any(np.all(symmetrical_matrix == 0, axis=1)) and not np.any(np.all(symmetrical_matrix == 0, axis=0)):
            pass
        else:
            symmetrical_matrix[row, col] = 1
            symmetrical_matrix[col, row] = 1

        print(symmetrical_matrix)
        print(np.sum(symmetrical_matrix))

    return symmetrical_matrix


# 生成边数递增的矩阵序列
def edgeRise_matrix_seq(startmat,num_ones):
    matrix_seq = []
    current_num_ones = np.sum(startmat)
    print(current_num_ones)
    while np.sum(startmat) < num_ones:
        row = np.random.randint(len(startmat))
        myset = set(range(len(startmat)))
        myset.discard(row)
        col = random.choice(list(myset))
        print(row,col)
        if startmat[row, col] == 0:
            startmat[row, col] = 1
            startmat[col, row] = 1
        else:
            # 如果当前位置的元素是 1，则循环直到不为1
            while startmat[row, col] != 0:
                row = np.random.randint(len(startmat))
                myset = set(range(len(startmat)))
                myset.discard(row)
                col = random.choice(list(myset))
                if startmat[row, col] == 0:
                    startmat[row, col] = 1
                    startmat[col, row] = 1
                    break
        matrix_seq.append(startmat.copy())
        # print(np.sum(startmat))
    return matrix_seq

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
net3 = edgeRise_matrix_seq(adj_matrix3,600)
# print(net3)
elesum3 = [(0.5 * np.sum(ele)) for ele in net3]
print(elesum3)
print(len(elesum3))

np.save("seqnet_3.npy", net3)
np.save("net_1.npy", adj_matrix1)
np.save("net_4.npy", adj_matrix4)
np.save("net_2.npy", adj_matrix2)



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# 对角线元素赋0值函数
def set_diagonal_to_zero(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    min_dim = min(rows, cols)
    for i in range(min_dim):
        matrix[i][i] = 0
    return matrix

# 生成无零行列对称矩阵
def generate_random_symmetric_binary_matrix_nozero(size):
    while True:
        upper_triangular = np.random.randint(2, size=(size, size))
        symmetrical_matrix = np.triu(upper_triangular) + np.triu(upper_triangular, 1).T
        symmetrical_matrix = set_diagonal_to_zero(symmetrical_matrix)

        # 检查每一行和每一列是否都不全为0
        if not np.any(np.all(symmetrical_matrix == 0, axis=1)) and not np.any(np.all(symmetrical_matrix == 0, axis=0)):
            return symmetrical_matrix

# 增加一个节点函数
def nodePlus(matrix):
    dim = len(matrix)
    new_column = np.random.randint(0, 2, size=(1, dim))
    new_column = new_column[0]
    while True:
        if np.sum(new_column) != 0:
            break
        else:
            new_column = np.random.randint(0, 2, size=(1, dim))

    new_matrix = np.hstack((matrix, new_column[:, np.newaxis]))
    new_row = np.append(new_column,0)
    new_matrix = np.vstack((new_matrix,new_row))
    return new_matrix

# matrix为初始矩阵，num代表增加节点数的步数
def nodeIncreaseSeq(matrix,num):
    startnode = len(matrix)
    startedge = 0.5 * np.sum(matrix)
    dim = len(matrix) + num
    startmat = matrix
    startmatcol = np.zeros((len(startmat), dim - len(startmat)))
    startmatrow = np.zeros((dim - len(startmat), dim))
    res_step1 = np.hstack((startmat, startmatcol))
    res_step2 = np.vstack((res_step1, startmatrow))

    mat_seq = [res_step2]
    node_seq = [startnode]
    edge_seq = [startedge]

    dic = {}
    i = 1
    while i <= num:
        endmat = nodePlus(startmat)
        matcol = np.zeros((len(endmat),dim-len(endmat)))
        matrow = np.zeros((dim-len(endmat),dim))
        res1 = np.hstack((endmat,matcol))
        res2 = np.vstack((res1,matrow))
        mat_seq.append(res2)
        node_seq.append(len(endmat))
        edge_seq.append(0.5 * np.sum(endmat))
        startmat = endmat
        i += 1

    dic['seq'] = mat_seq
    dic['nodes'] = node_seq
    dic['edges'] = edge_seq
    return dic


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



# 邻接矩阵序列1，节点数从10升到60
# 邻接矩阵序列2，节点数从20升到70
# 邻接矩阵序列3，节点数从30升到80
# 邻接矩阵序列4，节点数从10升到20
net4 = nodeIncreaseSeq(adj_matrix4,50)

elesum_seq = net4['seq']
print([len(ele) for ele in elesum_seq])
elesum_nodes = net4['nodes']
# print(elesum_seq)
print(elesum_nodes)
print(len(elesum_nodes))

np.save("node_net_4.npy", elesum_nodes)
np.save("seqnet_4.npy", elesum_seq)
np.save("net_1.npy", adj_matrix1)
np.save("net_2.npy", adj_matrix2)
np.save("net_3.npy", adj_matrix3)

import numpy as np
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt

# 方阵单层扩张函数
def matPluse(matrix,dim):
    if len(matrix) == dim:
        return matrix
    else:
        rownum = len(matrix)
        colnumup = dim - rownum
        matup = np.zeros((rownum, colnumup))
        matdown = np.zeros((colnumup, dim))
        matrix_new = np.hstack((matrix,matup))
        matrix_new = np.vstack((matrix_new,matdown))
        return matrix_new

def matSeqIncrease(mat_seq,dim):
    new_matseq = [matPluse(ele,dim) for ele in mat_seq]
    return new_matseq

# 层间超边扩张函数

def supermatPluse(matrix,dim_row,dim_col):
    rownum = matrix.shape[0]
    colnum = matrix.shape[1]
    if rownum < dim_row and colnum < dim_col:
        matup = np.zeros((rownum, dim_col - colnum))
        matdown = np.zeros((dim_row - rownum, dim_col))
        matrix_new = np.hstack((matrix,matup))
        matrix_new = np.vstack((matrix_new,matdown))
        return matrix_new
    elif rownum < dim_row and colnum == dim_col:
        matdown = np.zeros((dim_row - rownum, dim_col))
        matrix_new = np.vstack((matrix, matdown))
        return matrix_new
    elif rownum == dim_row and colnum < dim_col:
        matup = np.zeros((rownum, dim_col - colnum))
        matrix_new = np.hstack((matrix, matup))
        return matrix_new
    elif rownum == dim_row and colnum == dim_col:
        return matrix

def supermatSeqIncrease(mat_seq,dim_row,dim_col):
    new_matseq = [supermatPluse(ele,dim_row,dim_col) for ele in mat_seq]
    return new_matseq



# 生成随机整数数组1
random_integers1 = np.random.randint(10, 110, size=3000)  # 生成包含100个随机整数的数组
print("随机整数数组1:", random_integers1)


# 生成随机整数数组2
random_integers2 = np.random.randint(20, 120, size=3000)  # 生成包含100个随机整数的数组
print("随机整数数组2:", random_integers2)


# 生成随机整数数组3
random_integers3 = np.random.randint(30, 130, size=3000)  # 生成包含100个随机整数的数组
print("随机整数数组3:", random_integers3)


# 生成随机整数数组4
random_integers4 = np.random.randint(40, 140, size=3000)  # 生成包含100个随机整数的数组
print("随机整数数组4:", random_integers4)



# 创建第一层的图
G1 = nx.Graph()
# 添加初始节点
G1 = nx.barabasi_albert_graph(10, random.randint(1,9))
print(G1.degree())

matrix1_seq = []
# 生成无标度网络序列
for i in random_integers1:
    G1 = nx.barabasi_albert_graph(i, random.randint(1, i-1))
    adj_matrix1 = nx.to_numpy_matrix(G1)
    print(adj_matrix1)
    matrix1_seq.append(copy.copy(adj_matrix1))

lenmat1 = [len(ele) for ele in matrix1_seq]
np.save('nodes1.npy',lenmat1)

print(matrix1_seq)
print(lenmat1)
print(len(lenmat1))

# 创建第二层的图
G2 = nx.Graph()
# 添加初始节点
G2 = nx.barabasi_albert_graph(20, random.randint(1,19))
print(G2.degree())

matrix2_seq = []
# 生成无标度网络序列
for i in random_integers2:
    G2 = nx.barabasi_albert_graph(i, random.randint(1, i-1))
    adj_matrix2 = nx.to_numpy_matrix(G2)
    # print(adj_matrix2)
    matrix2_seq.append(copy.copy(adj_matrix2))

lenmat2= [len(ele) for ele in matrix2_seq]
np.save('nodes2.npy',lenmat2)
print(lenmat2)
print(len(lenmat2))

# 创建第三层的图
G3 = nx.Graph()
# 添加初始节点
G3 = nx.barabasi_albert_graph(30, random.randint(1,29))
print(G3.degree())

matrix3_seq = []
# 生成无标度网络序列
for i in random_integers3:
    G3 = nx.barabasi_albert_graph(i, random.randint(1, i-1))
    adj_matrix3 = nx.to_numpy_matrix(G3)
    # print(adj_matrix2)
    matrix3_seq.append(copy.copy(adj_matrix3))

lenmat3= [len(ele) for ele in matrix3_seq]
np.save('nodes3.npy',lenmat3)
print(lenmat3)
print(len(lenmat3))


# 创建第四层的图
G4 = nx.Graph()
# 添加初始节点
G4 = nx.barabasi_albert_graph(40, random.randint(1,39))
print(G4.degree())

matrix4_seq = []
# 生成无标度网络序列
for i in random_integers4:
    G4 = nx.barabasi_albert_graph(i, random.randint(1, i-1))
    adj_matrix4 = nx.to_numpy_matrix(G4)
    # print(adj_matrix2)
    matrix4_seq.append(copy.copy(adj_matrix4))

lenmat4= [len(ele) for ele in matrix4_seq]
np.save('nodes4.npy',lenmat4)
print(lenmat4)
print(len(lenmat4))


dim1 = max(lenmat1)
dim2 = max(lenmat2)
dim3 = max(lenmat3)
dim4 = max(lenmat4)

print(dim1,dim2,dim3,dim4)

netseq1_2 = []
for i in range(len(matrix1_seq)):
    rownum = len(matrix1_seq[i])
    colnum = len(matrix2_seq[i])
    midmat = np.random.randint(2, size=(rownum, colnum))
    netseq1_2.append(copy.copy(midmat))

print(netseq1_2)

netseq1_3 = []
for i in range(len(matrix1_seq)):
    rownum = len(matrix1_seq[i])
    colnum = len(matrix3_seq[i])
    midmat = np.random.randint(2, size=(rownum, colnum))
    netseq1_3.append(copy.copy(midmat))

print(netseq1_3)

netseq1_4 = []
for i in range(len(matrix1_seq)):
    rownum = len(matrix1_seq[i])
    colnum = len(matrix4_seq[i])
    midmat = np.random.randint(2, size=(rownum, colnum))
    netseq1_4.append(copy.copy(midmat))

print(netseq1_4)

netseq2_3 = []
for i in range(len(matrix1_seq)):
    rownum = len(matrix2_seq[i])
    colnum = len(matrix3_seq[i])
    midmat = np.random.randint(2, size=(rownum, colnum))
    netseq2_3.append(copy.copy(midmat))

print(netseq2_3)


netseq2_4 = []
for i in range(len(matrix1_seq)):
    rownum = len(matrix2_seq[i])
    colnum = len(matrix4_seq[i])
    midmat = np.random.randint(2, size=(rownum, colnum))
    netseq2_4.append(copy.copy(midmat))

print(netseq2_4)


netseq3_4 = []
for i in range(len(matrix1_seq)):
    rownum = len(matrix3_seq[i])
    colnum = len(matrix4_seq[i])
    midmat = np.random.randint(2, size=(rownum, colnum))
    netseq3_4.append(copy.copy(midmat))

print(netseq3_4)

# ==============================创建层内网络====================================
print('=========================net in layer=============================')
layer1 = matSeqIncrease(matrix1_seq,dim1)
# print(layer1)
dimmat1 = [ele.shape for ele in layer1]
dimsum1 = [np.sum(ele) for ele in layer1]
print(dimmat1)
print(dimsum1)



layer2 = matSeqIncrease(matrix2_seq,dim2)
# print(layer2)
dimmat2 = [ele.shape for ele in layer2]
dimsum2 = [np.sum(ele) for ele in layer2]
print(dimmat2)
print(dimsum2)


layer3 = matSeqIncrease(matrix3_seq,dim3)
# print(layer3)
dimmat3 = [ele.shape for ele in layer3]
dimsum3 = [np.sum(ele) for ele in layer3]
print(dimmat3)
print(dimsum3)



layer4 = matSeqIncrease(matrix4_seq,dim4)
# print(layer4)
dimmat4 = [ele.shape for ele in layer4]
dimsum4 = [np.sum(ele) for ele in layer4]
print(dimmat4)
print(dimsum4)

print('=========================supernet=============================')
supernetseq1_2 = supermatSeqIncrease(netseq1_2,dim1,dim2)
dimsuper1_2 = [ele.shape for ele in supernetseq1_2]
dimsupersum1_2 = [np.sum(ele) for ele in supernetseq1_2]
print(dimsuper1_2)
print(dimsupersum1_2)

supernetseq1_3 = supermatSeqIncrease(netseq1_3,dim1,dim3)
dimsuper1_3 = [ele.shape for ele in supernetseq1_3]
dimsupersum1_3 = [np.sum(ele) for ele in supernetseq1_3]
print(dimsuper1_3)
print(dimsupersum1_3)

supernetseq1_4 = supermatSeqIncrease(netseq1_4,dim1,dim4)
dimsuper1_4 = [ele.shape for ele in supernetseq1_4]
dimsupersum1_4 = [np.sum(ele) for ele in supernetseq1_4]
print(dimsuper1_4)
print(dimsupersum1_4)

supernetseq2_3 = supermatSeqIncrease(netseq2_3,dim2,dim3)
dimsuper2_3 = [ele.shape for ele in supernetseq2_3]
dimsupersum2_3 = [np.sum(ele) for ele in supernetseq2_3]
print(dimsuper2_3)
print(dimsupersum2_3)

supernetseq2_4 = supermatSeqIncrease(netseq2_4,dim2,dim4)
dimsuper2_4 = [ele.shape for ele in supernetseq2_4]
dimsupersum2_4 = [np.sum(ele) for ele in supernetseq2_4]
print(dimsuper2_4)
print(dimsupersum2_4)

supernetseq3_4 = supermatSeqIncrease(netseq3_4,dim3,dim4)
dimsuper3_4 = [ele.shape for ele in supernetseq3_4]
dimsupersum3_4 = [np.sum(ele) for ele in supernetseq3_4]
print(dimsuper3_4)
print(dimsupersum3_4)


block_matrix_seq = [np.array([[layer1[i], supernetseq1_2[i], supernetseq1_3[i], supernetseq1_4[i]],
                             [np.transpose(supernetseq1_2[i]), layer2[i], supernetseq2_3[i], supernetseq2_4[i]],
                             [np.transpose(supernetseq1_3[i]), np.transpose(supernetseq2_3[i]), layer3[i], supernetseq3_4[i]],
                             [np.transpose(supernetseq1_4[i]), np.transpose(supernetseq2_4[i]), np.transpose(supernetseq3_4[i]), layer4[i]]]
                             )
                    for i in range(len(matrix1_seq))]

print(block_matrix_seq)
np.save('block_matrix_randomseq_2.npy',block_matrix_seq)
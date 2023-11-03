import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from IO_PageRankSimulation import IOSuperPageRank as IOS

loaded_net4 = np.load("seqnet_4.npy")
# print(loaded_net1)
elesum = np.load("node_net_4.npy")
# print(elesum)


loaded_net1 = np.load("net_1.npy")
# print(loaded_net2)
loaded_net2 = np.load("net_2.npy")
# print(loaded_net3)
loaded_net3 = np.load("net_3.npy")
# print(loaded_net4)

loaded_net2 = [loaded_net2] * len(elesum)
# print(loaded_net1)
loaded_net3 = [loaded_net3] * len(elesum)
# print(loaded_net4)
loaded_net1 = [loaded_net1] * len(elesum)
# print(loaded_net2)

# 超边矩阵序列

def colplus(matrix):
    dim = matrix.shape[1]  # 获取矩阵的列数
    new_column = np.random.randint(0, 2, size=(matrix.shape[0], 1))  # 创建与矩阵行数相同的列向量
    while True:
        if np.sum(new_column) != 0:
            break
        else:
            new_column = np.random.randint(0, 2, size=(matrix.shape[0], 1))

    new_matrix = np.hstack((matrix, new_column))  # 将列向量与矩阵水平拼接
    return new_matrix


def colIncrease(matrix,num):
    rowsnum = matrix.shape[0]
    colsnum = matrix.shape[1]

    dim = colsnum + num
    startmat = matrix
    startmatcol = np.zeros((rowsnum, num))

    res_step = np.hstack((startmat, startmatcol))
    mat_seq = [res_step]
    i = 1
    while i <= num:
        endmat = colplus(startmat)
        matcol = np.zeros((rowsnum, dim - endmat.shape[1]))
        res = np.hstack((endmat, matcol))
        mat_seq.append(res)
        startmat = endmat
        i += 1

    return mat_seq


def rowplus(matrix):
    dim = matrix.shape[1]  # 获取矩阵的列数
    new_row = np.random.randint(0, 2, size=(1, dim))  # 创建与矩阵列数相同的行向量
    while True:
        if np.sum(new_row) != 0:
            break
        else:
            new_row = np.random.randint(0, 2, size=(1, dim))

    new_matrix = np.vstack((matrix, new_row))  # 将行向量与矩阵垂直拼接
    return new_matrix

def rowIncrease(matrix,num):
    rowsnum = matrix.shape[0]
    colsnum = matrix.shape[1]

    dim = rowsnum + num
    startmat = matrix
    startmatrow = np.zeros((num, colsnum))

    res_step = np.vstack((startmat, startmatrow))
    mat_seq = [res_step]
    i = 1
    while i <= num:
        endmat = rowplus(startmat)
        matrow = np.zeros((dim - endmat.shape[0], colsnum))
        res = np.vstack((endmat, matrow))
        mat_seq.append(res)
        startmat = endmat
        i += 1

    return mat_seq

dim = elesum[-1] - elesum[0]
print(dim)

net1_4 = np.random.randint(2, size=(10, 40))
print(net1_4)
net1_4 = colIncrease(net1_4,dim)
print(len(net1_4))
print(net1_4[0].shape)
#
net2_4 = np.random.randint(2, size=(20, 40))
print(net2_4)
net2_4 = colIncrease(net2_4,dim)
print(len(net2_4))
print(net2_4[0].shape)
#
net3_4 = np.random.randint(2, size=(30, 40))
print(net3_4)
net3_4 = colIncrease(net3_4,dim)
print(len(net3_4))
print(net3_4[0].shape)
#
# ==========================================================================
net1_2  = np.random.randint(2, size=(10, 20))
print(net1_2)
net1_2 = [net1_2] * len(elesum)
print(len(net1_2))
print(net1_2[0].shape)

net1_3  = np.random.randint(2, size=(10, 30))
print(net1_3)
net1_3 = [net1_3] * len(elesum)
print(len(net1_3))
print(net1_3[0].shape)

net2_3  = np.random.randint(2, size=(20, 30))
print(net2_3)
net2_3 = [net2_3] * len(elesum)
print(len(net2_3))
print(net2_3[0].shape)



print(len(loaded_net1))
print(len(loaded_net2))
print(len(loaded_net3))
print(len(loaded_net4))


print(loaded_net1[0].shape)
print(loaded_net2[0].shape)
print(loaded_net3[0].shape)
print(loaded_net4[0].shape)
#
block_matrix_seq = [np.array([[loaded_net1[i], net1_2[i], net1_3[i], net1_4[i]],
                             [np.transpose(net1_2[i]), loaded_net2[i], net2_3[i], net2_4[i]],
                             [np.transpose(net1_3[i]), np.transpose(net2_3[i]), loaded_net3[i], net3_4[i]],
                             [np.transpose(net1_4[i]), np.transpose(net2_4[i]), np.transpose(net3_4[i]), loaded_net4[i]]]
                             )
                    for i in range(len(elesum))]

print(block_matrix_seq)
#
np.save('block_matrix_nodes4.npy',block_matrix_seq)

tempio_seq = []

# #
for i in range(1,len(block_matrix_seq)):
    tempio_value = IOS.IOSuperPageRank(block_matrix_seq[0],block_matrix_seq[i])
    tempio_seq.append(tempio_value)

print(tempio_seq)
seq = [ele.MIO() for ele in tempio_seq]
#
seq_nodeinlayer = [ele[3][3] for ele in seq]
print(seq_nodeinlayer)
np.save("node_inlayer4.npy", seq_nodeinlayer)

seq_node1 = [ele[0][3] for ele in seq]
print(seq_node1)
np.save("node1.npy", seq_node1)

seq_node2 = [ele[1][3] for ele in seq]
print(seq_node2)
np.save("node2.npy", seq_node2)

seq_node3 = [ele[2][3] for ele in seq]
print(seq_node3)
np.save("node3.npy", seq_node3)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from IO_PageRank import IOSuperPageRank as IOS
from IO_PageRank import SuperNetTest as TS


# 生成随机矩阵
def generate_random_binary_matrix(rows, cols):
    return np.random.randint(2, size=(rows, cols))

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


def generate_random_binary_matrix_givensize(size_row,size_col,num_ones):
    # Create an empty binary matrix of the given size
    symmetrical_matrix = np.zeros((size_row, size_col), dtype=int)

    # Fill the matrix with the desired number of ones
    if num_ones > 0:
        indices = np.random.choice(size_row * size_col, num_ones, replace=False)
        symmetrical_matrix.flat[indices] = 1

    return symmetrical_matrix


# 生成边数递增的矩阵序列
def edgeRise_matrix_seq(startmat, num_ones):
    matrix_seq = []
    current_num_ones = np.sum(startmat)
    print(current_num_ones)
    startmat = startmat
    while current_num_ones < num_ones:
        # Make a copy of the startmat to avoid modifying it in place
        newmat = startmat.copy()

        # Randomly select a position and set it to 1 if it's 0
        row, col = np.random.randint(newmat.shape[0]), np.random.randint(newmat.shape[1])
        if newmat[row, col] == 0:
            newmat[row, col] = 1
        else:
            while newmat[row, col] != 0:
                row, col = np.random.randint(newmat.shape[0]), np.random.randint(newmat.shape[1])
            newmat[row, col] = 1


        matrix_seq.append(newmat)
        current_num_ones = np.sum(newmat)
        startmat = newmat
        print(newmat)
        print(current_num_ones)

    return matrix_seq


# 超边矩阵序列

net3_4 = generate_random_binary_matrix_givensize(30,40,30)
print(net3_4.shape)
# print(net2_3)
print(np.sum(net3_4))
net3_4 = edgeRise_matrix_seq(net3_4,120)
elesum = [np.sum(net3_4[i]) for i in range(len(net3_4))]
print(elesum)
print(len(elesum))
np.save("superedgenums2_4.npy", elesum)



net1_2 = np.random.randint(2, size=(10, 20))
print(net1_2)
net1_2 = [net1_2] * len(elesum)

net1_3 = np.random.randint(2, size=(10, 30))
print(net1_3)
net1_3 = [net1_3] * len(elesum)

net1_4 = np.random.randint(2, size=(10, 40))
print(net1_4)
net1_4 = [net1_4] * len(elesum)

net2_3 = np.random.randint(2, size=(20, 30))
print(net2_3)
net2_3 = [net2_3] * len(elesum)

net2_4 = np.random.randint(2, size=(20, 40))
print(net2_4)
net2_4 = [net2_4] * len(elesum)

loaded_net1 = np.load("net_1.npy")
loaded_net1 = [loaded_net1] * len(elesum)
# print(loaded_net1)

loaded_net2 = np.load("net_2.npy")
loaded_net2 = [loaded_net2] * len(elesum)
# print(loaded_net2)

loaded_net3 = np.load("net_3.npy")
loaded_net3 = [loaded_net3] * len(elesum)
# print(loaded_net3)

loaded_net4 = np.load("net_4.npy")
loaded_net4 = [loaded_net4] * len(elesum)
# print(loaded_net4)

block_matrix_seq = [np.array([[loaded_net1[i], net1_2[i], net1_3[i], net1_4[i]],
                             [np.transpose(net1_2[i]), loaded_net2[i], net2_3[i], net2_4[i]],
                             [np.transpose(net1_3[i]), np.transpose(net2_3[i]), loaded_net3[i], net3_4[i]],
                             [np.transpose(net1_4[i]), np.transpose(net2_4[i]), np.transpose(net3_4[i]), loaded_net4[i]]]
                             )
                    for i in range(len(elesum))]

# print(block_matrix_seq)

np.save('block_matrix_superedge3_4.npy',block_matrix_seq)

tempio_seq = []

# #
for i in range(1,len(block_matrix_seq)):
    tempio_value = IOS.IOSuperPageRank(block_matrix_seq[0],block_matrix_seq[i])
    tempio_seq.append(tempio_value)

print(tempio_seq)
seq = [ele.MIO() for ele in tempio_seq]
#
superedge1 = [ele[2][3] for ele in seq]
print(superedge1)
np.save("superedge34_3.npy", superedge1)

superedge2 = [ele[3][2] for ele in seq]
print(superedge2)
np.save("superedge34_4.npy", superedge2)



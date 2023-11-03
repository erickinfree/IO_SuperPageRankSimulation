import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from IO_PageRankSimulation import IOSuperPageRank as IOS

loaded_net3 = np.load("seqnet_3.npy")
print(loaded_net3)
elesum = [(0.5 * np.sum(ele)) for ele in loaded_net3]
print(elesum)
print(len(elesum))
np.save("edge_net_1.npy", elesum)

loaded_net4 = np.load("net_4.npy")
print(loaded_net4)
loaded_net1 = np.load("net_1.npy")
print(loaded_net1)
loaded_net2 = np.load("net_2.npy")
print(loaded_net2)

loaded_net1 = [loaded_net1] * len(elesum)
print(loaded_net1)
loaded_net4 = [loaded_net4] * len(elesum)
print(loaded_net4)
loaded_net2 = [loaded_net2] * len(elesum)
print(loaded_net2)

# 超边序列
mat1_2 = np.random.randint(2, size=(10, 20))
print(mat1_2)
mat1_2 = [mat1_2] * len(elesum)

mat1_3 = np.random.randint(2, size=(10, 30))
print(mat1_3)
mat1_3 = [mat1_3] * len(elesum)

mat1_4 = np.random.randint(2, size=(10, 40))
print(mat1_4)
mat1_4 = [mat1_4] * len(elesum)

mat2_3 = np.random.randint(2, size=(20, 30))
print(mat2_3)
mat2_3 = [mat2_3] * len(elesum)

mat2_4 = np.random.randint(2, size=(20, 40))
print(mat2_4)
mat2_4 = [mat2_4] * len(elesum)

mat3_4 = np.random.randint(2, size=(30, 40))
print(mat3_4)
mat3_4 = [mat3_4] * len(elesum)


block_matrix_seq = [np.array([[loaded_net1[i], mat1_2[i], mat1_3[i], mat1_4[i]],
                             [np.transpose(mat1_2[i]), loaded_net2[i], mat2_3[i], mat2_4[i]],
                             [np.transpose(mat1_3[i]), np.transpose(mat2_3[i]), loaded_net3[i],mat3_4[i]],
                             [np.transpose(mat1_4[i]), np.transpose(mat2_4[i]), np.transpose(mat3_4[i]), loaded_net4[i]]])
                    for i in range(len(elesum))]

np.save('block_matrix_edgenet1.npy',block_matrix_seq)


print('================================================================')
tempio_seq = []
# tempio = IOS.IOSuperPageRank(block_matrix_seq[0],block_matrix_seq[50])
# print(tempio.MIO())
# #
for i in range(1,len(block_matrix_seq)):
    tempio_value = IOS.IOSuperPageRank(block_matrix_seq[0],block_matrix_seq[i])
    tempio_seq.append(tempio_value)

print(tempio_seq)
seq = [ele.MIO() for ele in tempio_seq]
#
seq_edgeinlayer = [ele[2][2] for ele in seq]
print(seq_edgeinlayer)
np.save("net3_inlayer.npy", seq_edgeinlayer)

seq_net1 = [ele[0][2] for ele in seq]
print(seq_net1)
np.save("seq_net1.npy", seq_net1)

seq_net4 = [ele[3][2] for ele in seq]
print(seq_net4)
np.save("seq_net4.npy", seq_net4)

seq_net2 = [ele[1][2] for ele in seq]
print(seq_net2)
np.save("seq_net2.npy", seq_net2)
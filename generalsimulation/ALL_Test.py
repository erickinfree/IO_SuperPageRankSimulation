import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from IO_Test import IOSuperPageRank as IOS
from IO_PageRankSimulation import SuperNetTest as TS

block_matrix_seq = np.load('block_matrix_randomseq_2.npy',allow_pickle=True)
MIO_seq= np.load("MIO_randomseq_2.npy")

dimblock = block_matrix_seq[0]
dimlayer1 = len(dimblock[0][0])
dimlayer2 = len(dimblock[1][1])
dimlayer3 = len(dimblock[2][2])
dimlayer4 = len(dimblock[3][3])

# 超边数

print('==========================超边数=============================')
supernet1_2 = [np.sum(ele[0][1]) for ele in block_matrix_seq]
print(supernet1_2)


supernet1_3 = [np.sum(ele[0][2]) for ele in block_matrix_seq]
print(supernet1_3)

supernet1_4 = [np.sum(ele[0][3]) for ele in block_matrix_seq]
print(supernet1_4)

supernet2_3 = [np.sum(ele[1][2]) for ele in block_matrix_seq]
print(supernet2_3)

supernet2_4 = [np.sum(ele[1][3]) for ele in block_matrix_seq]
print(supernet2_4)

supernet3_4 = [np.sum(ele[2][3]) for ele in block_matrix_seq]
print(supernet3_4)

# 边数

print('==========================边数=============================')
edgenum1 = [0.5*np.sum(ele[0][0]) for ele in block_matrix_seq]
print(edgenum1)

edgenum2 = [0.5*np.sum(ele[1][1]) for ele in block_matrix_seq]
print(edgenum2)

edgenum3 = [0.5*np.sum(ele[2][2]) for ele in block_matrix_seq]
print(edgenum3)

edgenum4 = [0.5*np.sum(ele[3][3]) for ele in block_matrix_seq]
print(edgenum4)


# 边数

print('==========================节点数=============================')
nodes1 = np.load('nodes1.npy')
print(nodes1)
nodes2 = np.load('nodes2.npy')
print(nodes2)
nodes3 = np.load('nodes3.npy')
print(nodes3)
nodes4 = np.load('nodes4.npy')
print(nodes4)


h_inlayer = []
h_betweenlayer = []
for i in range(len(MIO_seq)):
    adss = TS.MatrixTest(MIO_seq[i], [nodes1[i], nodes2[i], nodes3[i], nodes4[i]], 0.1, 0.05, 0.05, 'Hommel')
    # print(adss.InLayer())
    h_inlayer.append(adss.InLayer())
    # print(adss.BetweenLayer())
    h_betweenlayer.append(adss.BetweenLayer())
    # print(adss.Extream())
    print(i)

print(h_inlayer)
print(h_betweenlayer)
h_inlayer = pd.DataFrame({'h_inlayer':h_inlayer})
h_inlayer.to_csv('h_inlayer_0.1.csv', index=False)

h_betweenlayer = pd.DataFrame({'h_betweenlayer':h_betweenlayer})
h_betweenlayer.to_csv('h_betweenlayer_0.1.csv', index=False)

# print('======================================================')
# for i in range(len(MIO_seq)):
#     adss = TS.MatrixTest(MIO_seq[i], [nodes1[i], nodes2[i], nodes3[i], nodes4[i]], 0.1, 0.05, 0.05, 'Holm')
#     print(adss.InLayer())
#
# print('======================================================')
# for i in range(len(MIO_seq)):
#     adss = TS.MatrixTest(MIO_seq[i], [nodes1[i], nodes2[i], nodes3[i], nodes4[i]], 0.6, 0.05, 0.05, 'Holm')
#     print(adss.InLayer())
# data1 = pd.DataFrame({
#     "h_inlayer": h_inlayer,
#     "nodes1": nodes1[1:],
#     "nodes2": nodes2[1:],
#     "nodes3": nodes3[1:],
#     "nodes4": nodes4[1:],
#     "edge1":edgenum1[1:],
#     "edge2":edgenum2[1:],
#     "edge3":edgenum3[1:],
#     "edge4":edgenum4[1:],
#     "superedge12":supernet1_2[1:],
#     "superedge13":supernet1_3[1:],
#     "superedge14":supernet1_4[1:],
#     "superedge23":supernet2_3[1:],
#     "superedge24":supernet2_4[1:],
#     "superedge34":supernet3_4[1:]
# })
#
# print(data1)
# data1.to_csv('datainlayer.csv', index=False)
#
# data2 = pd.DataFrame({
#     "h_betweenlayer": h_betweenlayer,
#     "nodes1": nodes1[1:],
#     "nodes2": nodes2[1:],
#     "nodes3": nodes3[1:],
#     "nodes4": nodes4[1:],
#     "edge1":edgenum1[1:],
#     "edge2":edgenum2[1:],
#     "edge3":edgenum3[1:],
#     "edge4":edgenum4[1:],
#     "superedge12":supernet1_2[1:],
#     "superedge13":supernet1_3[1:],
#     "superedge14":supernet1_4[1:],
#     "superedge23":supernet2_3[1:],
#     "superedge24":supernet2_4[1:],
#     "superedge34":supernet3_4[1:]
# })
#
# print(data2)
# data2.to_csv('databetweenlayer.csv', index=False)
#
# data3 = pd.DataFrame({
#     "MIO44": [ele[3][3] for ele in MIO_seq],
#     "nodes1": nodes1[1:],
#     "nodes2": nodes2[1:],
#     "nodes3": nodes3[1:],
#     "nodes4": nodes4[1:],
#     "edge1":edgenum1[1:],
#     "edge2":edgenum2[1:],
#     "edge3":edgenum3[1:],
#     "edge4":edgenum4[1:],
#     "superedge12":supernet1_2[1:],
#     "superedge13":supernet1_3[1:],
#     "superedge14":supernet1_4[1:],
#     "superedge23":supernet2_3[1:],
#     "superedge24":supernet2_4[1:],
#     "superedge34":supernet3_4[1:]
# })
#
# print(data3)
# data3.to_csv('MIO44.csv', index=False)

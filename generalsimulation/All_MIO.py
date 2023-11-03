import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from IO_PageRankSimulation import IOSuperPageRank as IOS


block_matrix_seq = np.load("block_matrix_randomseq_2.npy",allow_pickle=True)

tempio_seq = []

# #
for i in range(1,len(block_matrix_seq)):
    tempio_value = IOS.IOSuperPageRank(block_matrix_seq[0],block_matrix_seq[i])
    tempio_seq.append(tempio_value)

print(tempio_seq)
seq = [ele.MIO() for ele in tempio_seq]
#
print(seq)

np.save('MIO_randomseq_2.npy',seq)
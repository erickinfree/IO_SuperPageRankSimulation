import pandas as pd
import networkx as nx
import numpy as np

# 求解逆序数
class InverseOrder:
    def __init__(self,arr):
        self.arr = arr

    def merge_and_count(self, left, mid, right):
        arr = self.arr
        left_subarray = arr[left:mid + 1]
        right_subarray = arr[mid + 1:right + 1]

        inversions = 0
        i = j = 0
        k = left

        while i < len(left_subarray) and j < len(right_subarray):
            if left_subarray[i] <= right_subarray[j]:
                arr[k] = left_subarray[i]
                i += 1
            else:
                arr[k] = right_subarray[j]
                j += 1
                # Count inversions when an element from right_subarray is picked.
                inversions += (mid + 1) - (left + i)
            k += 1

        while i < len(left_subarray):
            arr[k] = left_subarray[i]
            i += 1
            k += 1

        while j < len(right_subarray):
            arr[k] = right_subarray[j]
            j += 1
            k += 1

        return inversions

    def merge_sort_and_count(self, left, right):
        arr = self.arr
        inversions = 0
        if left < right:
            mid = (left + right) // 2

            inversions += self.merge_sort_and_count(left, mid)
            inversions += self.merge_sort_and_count(mid + 1, right)
            inversions += self.merge_and_count(left, mid, right)

        return inversions

    def count_inversions(self):
        arr = self.arr
        return self.merge_sort_and_count(0, len(arr) - 1)


class LayerInfluenceRank:

    def __init__(self,net1,net2,netcov):
        self.net1 = net1
        self.net2 = net2
        self.netcov = netcov

        # 通过邻接矩阵构建两个图对象

    def IST(self):
        # 从邻接矩阵创建有向图对象
        graph1 = nx.from_numpy_array(self.net1, create_using=nx.Graph)
        # 计算网络1的各节点pr值
        pagerank_list1 = nx.pagerank(graph1, alpha=0.85)
        dtResult1 = pd.DataFrame(columns=['key', 'valuestart'])
        for key in pagerank_list1:
            print(key, pagerank_list1[key])
            dtResult1.loc[len(dtResult1.index)] = [key, pagerank_list1[key]]
        print(dtResult1)
        # 计算网络1中各节点影响范围
        rowSum = np.sum(self.netcov,axis=1)
        print(rowSum)
        res = list(dtResult1['valuestart'])*rowSum
        return res

    def LIR(self):
        ISt = np.array(self.IST())
        res = np.dot(ISt,self.netcov)
        return res



if __name__ == '__main__':
    # 示例邻接矩阵
    adj_matrix1 = np.array([[0, 1, 1, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 1, 1, 0]])
    adj_matrix2 = np.array([[0, 1, 0, 0, 1],
                            [1, 0, 1, 1, 0],
                            [0, 1, 0, 1, 1],
                            [0, 1, 1, 0, 0],
                            [1, 0, 1, 0, 0]])
    adj_cov = np.array([[1, 1, 1, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 1, 0, 1],
                        [1, 0, 0, 1, 1]])

    a = LayerInfluenceRank(adj_matrix1,adj_matrix2,adj_cov)
    print(a.IST())
    print(a.LIR())

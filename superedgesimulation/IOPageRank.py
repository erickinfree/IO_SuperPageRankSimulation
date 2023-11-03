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



# 求解IOpagerank
class IOPageRank:
    # 输入两个网络
    def __init__(self,net1,net2):
        self.net1 = net1
        self.net2 = net2

    # 通过邻接矩阵构建两个图对象
    def Netting(self):
        # 从邻接矩阵创建有向图对象
        graph1 = nx.from_numpy_array(self.net1, create_using=nx.Graph)
        # 从邻接矩阵创建有向图对象
        graph2 = nx.from_numpy_array(self.net2, create_using=nx.Graph)
        # 返回包含前后两个时间的图
        return [graph1,graph2]

    # 通过两个图对象求解每张图上各点的PR值
    def PGR(self):
        # 对两张图求PR值，参数设为0.85
        pagerank_list1 = nx.pagerank(self.Netting()[0], alpha=0.85)
        pagerank_list2 = nx.pagerank(self.Netting()[1], alpha=0.85)

        # 将两张图的PR值分别放入dataframe
        dtResult1 = pd.DataFrame(columns=['key', 'valuestart'])
        dtResult2 = pd.DataFrame(columns=['key', 'valueend'])
        for key in pagerank_list1:
            print(key, pagerank_list1[key])
            dtResult1.loc[len(dtResult1.index)] = [key, pagerank_list1[key]]
        print(dtResult1)
        for key in pagerank_list2:
            print(key, pagerank_list2[key])
            dtResult2.loc[len(dtResult2.index)] = [key, pagerank_list2[key]]
        print(dtResult2)

        # 通过key将两张图归并
        merged_df = pd.merge(dtResult1, dtResult2, on='key', how='outer').fillna(0)
        print(merged_df)
        return merged_df

    # 按照第一列排序
    def StartSort(self):
        df = self.PGR()
        sorted_df = df.sort_values(by='valuestart')
        print(sorted_df)
        return sorted_df

    # 按照第二列求逆序
    def AntiSort(self):
        df = self.StartSort()
        IO= InverseOrder(list(df['valueend']))
        print('逆序数为：')
        return IO.count_inversions()


if __name__ == '__main__':
    # 示例邻接矩阵
    adj_matrix1 = np.array([[0, 1, 1, 0],
                           [1, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0]])
    adj_matrix2 = np.array([[0, 1, 0, 0, 1],
                           [1, 0, 1, 1, 0],
                           [0, 1, 0, 1, 1],
                           [0, 1, 1, 0, 0],
                           [1, 0, 1, 0, 0]])

    a = IOPageRank(adj_matrix1,adj_matrix2)
    a1 = a.Netting()
    print(a1[0].nodes(),a1[0].edges())
    print(a1[1].nodes(),a1[1].edges())
    print('-----------------------------------------------')
    a2 = a.StartSort()
    print(a2)
    print('-----------------------------------------------')
    a3 = a.AntiSort()
    print(a3)


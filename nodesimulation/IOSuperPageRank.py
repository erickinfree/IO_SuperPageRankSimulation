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
    # 计算net1对net2影响
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
            dtResult1.loc[len(dtResult1.index)] = [key, pagerank_list1[key]]
        # 计算网络1中各节点影响范围
        rowSum = np.sum(self.netcov,axis=1)
        # print(rowSum)
        res = list(dtResult1['valuestart'])*rowSum
        return res

    def LIR(self):
        ISt = np.array(self.IST())
        res = np.dot(ISt,self.netcov)
        return res

class IOSuperPageRank:

    # 输入两个超网络的矩阵
    def __init__(self,supermat1,supermat2):
        self.supermat1 = supermat1
        self.supermat2 = supermat2

    # 注意两期矩阵以左上方为公共交集，例如1期节点少于2期，以1期为准
    def LIRVect(self):

        # 矩阵1的维度
        num1 = len(self.supermat1)

        # 空矩阵装超网络1的各层LIR
        LIRmat1 = []

        for i in range(num1):
            # 装第i层网络的LIR
            rowLIR = []
            # 第i层受各层的影响
            for j in range(num1):
                # 第i层网络受本层的影响拿PR值代替
                if i == j:
                    graph = nx.from_numpy_array(self.supermat1[i][j], create_using=nx.Graph)
                    pagerank_list = nx.pagerank(graph, alpha=0.85)
                    dtResult = pd.DataFrame(columns=['key', 'valuestart'])
                    for key in pagerank_list:
                        dtResult.loc[len(dtResult.index)] = [key, pagerank_list[key]]
                    # print(dtResult)
                    rowLIR.append(dtResult)
                # 第i层受其他层影响利用LayerInfluence算法计算
                else:
                    LIRele = LayerInfluenceRank(self.supermat1[j][j],self.supermat1[i][i],self.supermat1[j][i])
                    resLIR = LIRele.LIR()
                    dtResult_r = pd.DataFrame(columns=['key', 'valuestart'])
                    for k in range(len(resLIR)):
                        dtResult_r.loc[len(dtResult_r.index)] = [k, resLIR[k]]
                    # print(dtResult_r)
                    rowLIR.append(dtResult_r)
            LIRmat1.append(rowLIR)
        LIRmat1 = np.array(LIRmat1)

        # 矩阵2的维度
        num2 = len(self.supermat2)

        # 空矩阵装超网络2的各层LIR
        LIRmat2 = []

        for i in range(num2):
            # 装第i层网络的LIR
            rowLIR = []
            # 第i层受各层的影响
            for j in range(num2):
                # 第i层网络受本层的影响拿PR值代替
                if i == j:
                    graph = nx.from_numpy_array(self.supermat2[i][j], create_using=nx.Graph)
                    pagerank_list = nx.pagerank(graph, alpha=0.85)
                    dtResult = pd.DataFrame(columns=['key', 'valueend'])
                    for key in pagerank_list:
                        dtResult.loc[len(dtResult.index)] = [key, pagerank_list[key]]
                    # print(dtResult)
                    rowLIR.append(dtResult)
                # 第i层受其他层影响利用LayerInfluence算法计算
                else:
                    LIRele = LayerInfluenceRank(self.supermat2[j][j], self.supermat2[i][i], self.supermat2[j][i])
                    resLIR = LIRele.LIR()
                    dtResult_r = pd.DataFrame(columns=['key', 'valueend'])
                    for k in range(len(resLIR)):
                        dtResult_r.loc[len(dtResult_r.index)] = [k, resLIR[k]]
                    # print(dtResult_r)
                    rowLIR.append(dtResult_r)
            LIRmat2.append(rowLIR)
        LIRmat2 = np.array(LIRmat2)

        return [LIRmat1,LIRmat2]

    # 计算两期之间的稳定性矩阵
    def MIO(self):
        res = self.LIRVect()
        mat1 = res[0]
        mat2 = res[1]
        MIOmat = []
        num = len(mat1)
        for i in range(num):
            rowMIO = []
            for j in range(num):
                merged_df = pd.merge(mat1[i][j], mat2[i][j], on='key', how='outer').fillna(0)
                sorted_df = merged_df.sort_values(by='valuestart')
                IO = InverseOrder(list(sorted_df['valueend'])).count_inversions()
                rowMIO.append(IO)
            MIOmat.append(rowMIO)
        return np.array(MIOmat)



        





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
    adj_matrix3 = np.array([[0, 1, 0, 0, 1, 1],
                            [1, 0, 1, 1, 0, 1],
                            [0, 1, 0, 1, 1, 1],
                            [0, 1, 1, 0, 1, 0],
                            [1, 0, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0, 0]])

    adj_cov12 = np.array([[1, 1, 1, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 1, 0, 1],
                        [1, 0, 0, 1, 1]])
    adj_cov23 = np.array([[1, 1, 1, 1, 0, 1],
                          [1, 0, 1, 0, 1, 0],
                          [0, 1, 1, 0, 1, 1],
                          [1, 0, 0, 1, 1, 0],
                          [1, 0, 0, 1, 1, 0]])
    adj_cov13 = np.array([[1, 0, 1, 1, 0, 1],
                          [1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 0, 1, 1],
                          [1, 0, 0, 0, 1, 0]])

    block_matrix1 = np.array([[adj_matrix1, adj_cov12, adj_cov13],
                             [np.transpose(adj_cov12), adj_matrix2, adj_cov23],
                             [np.transpose(adj_cov13), np.transpose(adj_cov23), adj_matrix3]])
    print(len(block_matrix1))

    adj_matrix1_p = np.array([[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [1, 1, 0, 1],
                            [1, 1, 1, 0]])
    adj_matrix2_p = np.array([[0, 1, 0, 0, 1, 0],
                            [1, 0, 1, 1, 0, 1],
                            [0, 1, 0, 1, 1, 0],
                            [0, 1, 1, 0, 0, 0],
                            [1, 0, 1, 0, 0, 1],
                            [0, 1, 0, 0, 1, 0]])
    adj_matrix3_p = np.array([[0, 0, 1, 0, 1, 1],
                            [0, 0, 1, 1, 0, 1],
                            [1, 1, 0, 0, 1, 1],
                            [0, 1, 0, 0, 1, 0],
                            [1, 0, 1, 1, 0, 1],
                            [1, 1, 1, 0, 1, 0]])

    adj_cov12_p = np.array([[1, 1, 1, 1, 0, 0],
                          [1, 0, 1, 0, 1, 1],
                          [0, 1, 1, 0, 1, 0],
                          [1, 0, 0, 1, 1, 1]])
    adj_cov23_p = np.array([[1, 1, 1, 1, 0, 1],
                          [1, 0, 1, 0, 1, 0],
                          [0, 1, 1, 0, 1, 1],
                          [1, 0, 0, 1, 1, 0],
                          [1, 0, 1, 0, 1, 0],
                          [1, 1, 0, 1, 1, 0]])
    adj_cov13_p = np.array([[1, 0, 1, 1, 0, 1],
                          [1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 0, 1, 1],
                          [1, 0, 0, 0, 1, 0]])

    block_matrix1_p = np.array([[adj_matrix1_p, adj_cov12_p, adj_cov13_p],
                              [np.transpose(adj_cov12_p), adj_matrix2_p, adj_cov23_p],
                              [np.transpose(adj_cov13_p), np.transpose(adj_cov23_p), adj_matrix3_p]])
    print(len(block_matrix1_p))


    a = IOSuperPageRank(block_matrix1,block_matrix1_p)
    print(a.LIRVect())
    print(a.MIO())
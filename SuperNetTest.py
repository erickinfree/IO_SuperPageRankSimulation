import numpy as np
import scipy as sp
import math
from statsmodels.stats.multitest import multipletests

class IOtest:
    def __init__(self,IO,N,p,a):
        self.IO = IO
        self.num = sp.special.comb(N,2)
        self.p = p
        self.a = a

    def IOTest(self):
        # print(self.num)
        factor1 = self.IO - (self.p * self.num)
        factor2 = math.sqrt(self.p * (1 - self.p) * self.num)
        Z = factor1/factor2
        # print(Z)
        p_value = 1 - sp.stats.norm.cdf(Z,loc = 0,scale = 1)
        if p_value < self.a:
            return [False,p_value]
        else:
            return [True,p_value]

    def IObiTest(self):
        # 注意，后续版本的spicy中函数为binomtest()
        p_value = sp.stats.binom_test(self.IO, n = self.num, p = self.p, alternative='greater')
        if p_value < self.a:
            return [False,p_value]
        else:
            return [True,p_value]

class multitest:
    def __init__(self,p_list,a,method_choice):
        self.p_list = p_list
        self.a = a
        self.method_choice = method_choice

    def Bonferroni(self):
        reject, p_corrected, _, _ = multipletests(self.p_list, alpha=self.a, method='bonferroni')
        V = sum(reject) / len(reject)
        return V

    def Holm(self):
        reject, p_corrected, _, _ = multipletests(self.p_list,alpha=self.a,method='holm')
        V = sum(reject)/len(reject)
        return V

    def BH(self):
        reject, p_corrected, _, _ = multipletests(self.p_list, alpha=self.a, method='fdr_bh')
        V = sum(reject) / len(reject)
        return V

    def Sidak(self):
        reject, p_corrected, _, _ = multipletests(self.p_list, alpha=self.a, method='sidak')
        V = sum(reject) / len(reject)
        return V

    def BY(self):
        reject, p_corrected, _, _ = multipletests(self.p_list, alpha=self.a, method='fdr_by')
        V = sum(reject) / len(reject)
        return V

    def Hommel(self):
        reject, p_corrected, _, _ = multipletests(self.p_list, alpha=self.a, method='hommel')
        V = sum(reject) / len(reject)
        return V

    def Choice(self):
        if self.method_choice == 'Holm':
            V = self.Holm()
            return V
        elif self.method_choice == 'Bonferroni':
            V = self.Bonferroni()
            return V
        elif self.method_choice == 'BH':
            V = self.BH()
            return V
        elif self.method_choice == 'Sidak':
            V = self.Sidak()
            return V
        elif self.method_choice == 'BY':
            V = self.BY()
            return V
        elif self.method_choice == 'Hommel':
            V = self.Hommel()
            return V

class MatrixTest:
    def __init__(self,MIO,d_list,p,a,mul_a,method_choice):
        self.MIO = MIO
        # 各层节点数
        self.d_list = d_list
        self.p = p
        self.a = a
        self.mul_a = mul_a
        self.method_choice = method_choice

    def InLayer(self):
        main_diagonal = np.diagonal(self.MIO)
        num = len(main_diagonal)
        p_list = []
        for i in range(num):
            diagtest = IOtest(main_diagonal[i],self.d_list[i],self.p,self.a)
            p_list.append(diagtest.IOTest()[1])
        test = multitest(p_list,self.mul_a,self.method_choice)
        h_in_layer = test.Choice()
        return h_in_layer

    def BetweenLayer(self):
        num = len(self.MIO)
        B_testlist = []
        for i in range(num):
            for j in range(num):
                if i != j:
                    nodiagtest = IOtest(self.MIO[i][j], self.d_list[i], self.p, self.a)
                    B_testlist.append(nodiagtest.IOTest()[1])
        test = multitest(B_testlist, self.mul_a,self.method_choice)
        h_between_layer =  test.Choice()
        return h_between_layer

    def Extream(self):
        max_element = np.max(self.MIO)
        return max_element





    






if __name__ == '__main__':
    a = IOtest(500,40,0.6,0.05)
    print(a.IOTest())
    print(a.IObiTest())

    b = multitest([0.01,0.02,0.08,0.09],0.05,'Holm')
    print(b.Choice())



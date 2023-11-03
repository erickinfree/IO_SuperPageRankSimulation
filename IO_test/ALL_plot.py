import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# 创建一个示例数据集，这里使用一个随机生成的矩阵
data = pd.read_excel('h_betweenlayer.xlsx')
print(data.iloc[:,1:])

# 创建一个热力图
# plt.figure(figsize=(14, 4))

cate = sns.heatmap(data.iloc[:,1:],
                   annot=True,
                   linewidths=0.5,
                   fmt=".3f",
                   cmap="YlGnBu")

row_labels = data['name']
cate.set_yticklabels(row_labels, rotation=0)

#
# # 添加标签和标题
# plt.xlabel("Year")
# plt.ylabel("Month")
# plt.title("Passenger Count Heatmap")

# 显示热力图
plt.tight_layout()
plt.savefig('h_betweenlayer.svg', format='svg')
plt.show()

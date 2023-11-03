import numpy as np
import matplotlib.pyplot as plt

node_num = np.load("node_net_3.npy")

net1 = np.load("node1.npy")
net2 = np.load("node2.npy")
net3 = np.load("node_inlayer3.npy")
net4 = np.load("node4.npy")


print(node_num)
print(net1)
print(net2)
print(net3)
print(net4)

# # 示例数据
x = [i for i in range(len(node_num))]
y = node_num

x1 = [i for i in range(len(net1))]
y1 = net1

x2 = [i for i in range(len(net2))]
y2 = net2

x3 = [i for i in range(len(net3))]
y3 = net3

x4 = [i for i in range(len(net4))]
y4 = net4


# 创建图形和轴对象
fig, ax1 = plt.subplots()

# 绘制第一个数据集和坐标轴（左侧）
ax1.plot(x, y, color='blue',
         marker='o',
         label='Number of Nodes in layer3',
         markersize=1)
ax1.set_xlabel('Period of Change')
ax1.set_ylabel('Number of Nodes', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 创建第二个坐标轴（右侧）
ax2 = ax1.twinx()
ax2.plot(x4, y4, color='purple',
         marker='x',
         label='IO[4][3]',
         markersize=1)
ax2.set_ylabel('Change in Inversion Number of the Layer4', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

#
# plt.ylim(72, 140)
# 显示图形
plt.savefig('layer4_3.svg', format='svg')
plt.show()

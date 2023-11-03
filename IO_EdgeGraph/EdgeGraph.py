import numpy as np
import matplotlib.pyplot as plt

edge_num = np.load("edge_net_4.npy")

net1 = np.load("seq_net1.npy")
net2 = np.load("seq_net2.npy")
net3 = np.load("seq_net3.npy")
net4 = np.load("net4_inlayer.npy")

print(edge_num)
print(len(edge_num))

print(net1)
print(len(net1))

print(net2)
print(len(net2))

print(net3)
print(len(net3))

print(net4)
print(len(net4))

# # 示例数据
x = [i for i in range(len(edge_num))]
y = edge_num

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
         label='Number of Edges in layer4',
         markersize=1)
ax1.set_xlabel('Period of Change')
ax1.set_ylabel('Number of Edges', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 创建第二个坐标轴（右侧）
ax2 = ax1.twinx()
ax2.plot(x4, y4, color='purple',
         marker='x',
         label='IO[4][4]',
         markersize=1)
ax2.set_ylabel('Change in Inversion Number of the Layer4', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

#
plt.ylim(0, 270)
# 显示图形
plt.savefig('layer4_4.svg', format='svg')
plt.show()



# # 创建图形和轴对象
# fig, ax2 = plt.subplots()
#
# # 绘制第一个数据集和坐标轴（左侧）
# ax1.plot(x, y, color='blue',
#          marker='o',
#          label='Number of Edges in layer2',
#          markersize=1)
# ax1.set_xlabel('Period of Change')
#
# ax1.set_ylabel('Number of Edges', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')
#
# # 创建第二个坐标轴（右侧）
# ax2 = ax1.twinx()
#
# # 创建第3个坐标轴（右侧）
# ax2.plot(x1, y1, color='red',
#          marker='x',
#          label='IO[1][4]',
#          markersize=1)
# ax2.set_ylabel('Change in Inversion Number of the Each Layer', color='red')
# ax2.tick_params(axis='y', labelcolor='red')
#
# # 创建第3个坐标轴（右侧）
# ax2.plot(x2, y2, color='green',
#          marker='x',
#          label='IO[2][2]',
#          markersize=1)
# ax2.set_ylabel('Change in Inversion Number of the Each Layer', color='green')
# ax2.tick_params(axis='y', labelcolor='green')
#
#
# # 创建第3个坐标轴（右侧）
# ax2.plot(x3, y3, color='orange',
#          marker='x',
#          label='IO[3][4]',
#          markersize=1)
# ax2.set_ylabel('Change in Inversion Number of the Layer2', color='orange')
# ax2.tick_params(axis='y', labelcolor='orange')
#
#
# # 创建第3个坐标轴（右侧）
# ax2.plot(x4, y4, color='purple',
#          marker='x',
#          label='IO[4][4]',
#          markersize=1)
# ax2.set_ylabel('Change in Inversion Number of Each Layer', color='purple')
# ax2.tick_params(axis='y', labelcolor='purple')
#
# plt.ylim(0, 350)
#
# # 添加图例
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper left')
# plt.savefig('Edgelayer4.svg', format='svg')
# plt.show()
#

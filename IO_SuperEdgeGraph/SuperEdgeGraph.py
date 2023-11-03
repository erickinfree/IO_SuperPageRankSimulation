import numpy as np
import matplotlib.pyplot as plt

superedge_num = np.load("superedgenums3_4.npy")
print(superedge_num[0])
print(superedge_num[-1])
# net1 = np.load("superedge34_3.npy")
# net2 = np.load("superedge34_4.npy")


# # # 示例数据
# x = [i for i in range(len(superedge_num))]
# y = superedge_num
#
# x1 = [i for i in range(len(net1))]
# y1 = net1
#
# x2 = [i for i in range(len(net2))]
# y2 = net2

# # 创建图形和轴对象
# fig, ax1 = plt.subplots()
#
# # 绘制第一个数据集和坐标轴（左侧）
# ax1.plot(x, y, color='blue',
#          marker='o',
#          label='Number of SuperEdges betwween layer3 and layer4',
#          markersize=1)
# ax1.set_xlabel('Period of Change')
# ax1.set_ylabel('Number of SuperEdges', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')
#
# plt.ylim(0, 140)
#
# # 创建第二个坐标轴（右侧）
# ax2 = ax1.twinx()
# ax2.plot(x1, y1, color='orange',
#          marker='x',
#          label='IO[3][4]',
#          markersize=1)
# ax2.set_ylabel('Change in Inversion Number of the Layer2', color='orange')
# ax2.tick_params(axis='y', labelcolor='orange')
#
# # 创建第二个坐标轴（右侧）
# ax2.plot(x2, y2, color='purple',
#          marker='x',
#          label='IO[4][3]',
#          markersize=1)
# ax2.set_ylabel('Change in Inversion Number of Layer3 and Layer4', color='black')
# ax2.tick_params(axis='y', labelcolor='black')
#
# # 添加图例
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper left')
#
# #
# plt.ylim(0, 350)
# # 显示图形
# plt.savefig('superedgegraph3_4.svg', format='svg')
# plt.show()

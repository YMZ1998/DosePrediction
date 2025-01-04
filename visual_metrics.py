import matplotlib.pyplot as plt
import numpy as np

# 数据
scores_before = [2.69, 1.67]  # 改进前的分数
scores_after = [1.93, 1.51]   # 改进后的分数
labels = ['Dose score', 'DVH score']

# 设置条形图的位置
x = np.arange(len(labels))  # 标签的位置
width = 0.35  # 每个条形的宽度

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制改进前后的条形图
rects1 = ax.bar(x - width/2, scores_before, width, label='Before', color='green', alpha=0.5)
rects2 = ax.bar(x + width/2, scores_after, width, label='After', color='blue', alpha=0.5)

# 添加一些文本标签
ax.set_ylabel('Scores')
ax.set_title('Comparison of Dose and DVH Scores Before and After Improvement')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 显示每个条形的高度
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)

# 显示图形
plt.tight_layout()
plt.savefig('visual_metrics.png', transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()


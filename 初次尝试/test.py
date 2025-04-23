import matplotlib.pyplot as plt
import numpy as np

x = np.arange(100)
y = np.sin(x / 10)

# 示例点
peaks = [20, 40, 70]
labels = [0, 1, 2]
cmap = plt.cm.get_cmap('tab10', 3)

plt.plot(x, y, color='blue')
for i, peak in enumerate(peaks):
    plt.plot(peak, y[peak], 'o', color=cmap(labels[i]), markersize=10, markeredgecolor='black')
    print(labels[i])
    print(cmap(labels[i]))

plt.title("颜色检查")
plt.show()
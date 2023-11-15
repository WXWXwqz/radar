import numpy as np
import matplotlib.pyplot as plt

# 假设你有一个包含点坐标的列表，每个点的坐标为(x, y)
points = [(x, y) for x in range(5000, 30001) for y in range(-3000, 501)]

# 创建一个二维数组表示图像，初始化为零
image = np.zeros((33001, 3501), dtype=np.uint8)

# 在图像中增加每个点的出现次数
for x, y in points:
    image[x, y] += 1

# 显示图像
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.savefig(f"./fig/tst.png")
# plt.show()
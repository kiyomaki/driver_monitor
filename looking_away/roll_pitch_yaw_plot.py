import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# プロットの準備
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 原点
origin = np.array([0, 0, 0])

# X, Y, Z軸方向のベクトル
vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
colors = ['r', 'g', 'b']

# ベクトルをプロット
for vector, color in zip(vectors, colors):
    ax.quiver(*origin, *vector, color=color)

# 軸ラベルの設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# プロット範囲の設定
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# プロットの表示
plt.show()


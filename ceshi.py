import numpy as np


def k_means(points, k):
    # 随机初始化聚类中心
    centers = points[np.random.choice(len(points), k, replace=False)]

    while True:
        # 分配每个点到最近的聚类中心
        labels = np.argmin(np.sum(np.square(points[:, np.newaxis, :] - centers), axis=2), axis=1)
        # 更新聚类中心
        new_centers = np.array([points[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return labels


# 创建 9 个点的坐标数组
points = np.array([[0, 8],[2, 8],[1, 9], [5, 6], [4, 7],  [4, 9], [7, 7], [6, 7],[5, 10]])

# 将 9 个点分成 3 组
labels = k_means(points, 3)
print(labels) # 输出每个点所属的组别

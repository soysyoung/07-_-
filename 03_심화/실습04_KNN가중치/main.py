"""실습 04 심화: 이전 응용의 거리 가중치를 표로 설명합니다."""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

X = np.array([[1], [2], [4], [7], [9]])
y = np.array([10, 20, 40, 70, 90])
plain = KNeighborsRegressor(n_neighbors=3).fit(X, y)
distance, indices = plain.kneighbors([[3.2]])
print(pd.DataFrame({"input": X[indices[0], 0], "distance": distance[0], "target": y[indices[0]]}).round(3).to_string(index=False))
# TODO H04: 가까운 이웃에 큰 가중치를 주도록 변경하세요.
# 힌트: weights="distance"
# weighted = KNeighborsRegressor(n_neighbors=3, weights="uniform")
# weights="distance"는 가까운 이웃일수록 예측값에 더 큰 영향을 주도록 설정합니다.
# 각 이웃의 가중치는 거리의 역수(1 / 거리)에 비례하므로, 거리가 짧을수록 가중치가 커집니다.
# 3.2와 가장 가까운 입력 4의 정답 40이 다른 정답보다 더 크게 반영됩니다.
# 아래의 np.average 손계산도 1 / distance를 사용하므로 같은 원리로 계산됩니다.
weighted = KNeighborsRegressor(n_neighbors=3, weights="distance")

weighted.fit(X, y)
print("uniform:", plain.predict([[3.2]]).round(2))
print("distance:", weighted.predict([[3.2]]).round(2))
print("역거리 가중 손계산:", round(np.average(y[indices[0]], weights=1 / distance[0]), 2))

"""실습 04 응용: 같은 세 이웃에서 가중치만 바꿉니다."""
from sklearn.neighbors import KNeighborsRegressor

X = [[1], [2], [4], [7], [9]]
y = [10, 20, 40, 70, 90]
for weight in ["uniform", "distance"]:
    # TODO A04: weights에 반복 변수 weight를 넣으세요.
    # 힌트: 이웃 수는 3으로 고정합니다.
    model = KNeighborsRegressor(n_neighbors=3, weights="uniform")
    # 첫 반복에서는 weight가 "uniform", 두 번째 반복에서는 "distance"가 됩니다.
    # uniform은 가까운 세 이웃의 정답을 모두 같은 비중으로 평균냅니다.
    # distance는 거리가 가까운 이웃일수록 더 큰 비중을 주어 평균냅니다.
    # 고정 문자열 대신 반복 변수 weight를 넣어 두 방식을 차례로 비교합니다.
    model = KNeighborsRegressor(n_neighbors=3, weights=weight)

    model.fit(X, y)
    print(weight, "예측:", model.predict([[3.2]]).round(2))
# uniform은 3개를 똑같이, distance는 가까운 값에 더 큰 비중을 줍니다.

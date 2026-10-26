"""실습 04 기본필수: 이웃 3개의 투표와 평균을 비교합니다."""
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

X = [[1], [2], [4], [7], [9]]
class_y = [0, 0, 1, 1, 1]
number_y = [10, 20, 40, 70, 90]
# 3.2에 가까운 입력은 4, 2, 1입니다. 거리: 0.8, 1.2, 2.2
# TODO B04a: 분류에서 이웃 수를 3으로 바꾸세요.
# 힌트: n_neighbors는 참고할 이웃 수입니다.
classifier = KNeighborsClassifier(n_neighbors=1)
# KNN 분류는 새로운 값과 가장 가까운 이웃 3개의 정답을 확인합니다.
# 3.2와 가까운 입력은 4, 2, 1이고, 대응하는 클래스는 [1, 0, 0]입니다.
# 세 이웃의 다수결 결과가 0이므로 새로운 값 3.2를 클래스 0으로 예측합니다.
classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(X, class_y)
print("분류:", classifier.predict([[3.2]]))
# TODO B04b: 회귀에서도 이웃 수를 3으로 바꾸세요.
# 힌트: 같은 이웃의 숫자 정답 [40, 20, 10]을 평균냅니다.
regressor = KNeighborsRegressor(n_neighbors=1)
# KNN 회귀도 가장 가까운 이웃 3개를 찾지만, 투표 대신 숫자 정답의 평균을 구합니다.
# 가까운 세 이웃의 정답은 [40, 20, 10]이므로 예측값은 (40 + 20 + 10) / 3 = 23.33입니다.
regressor = KNeighborsRegressor(n_neighbors=3)

regressor.fit(X, number_y)
print("회귀:", regressor.predict([[3.2]]).round(2))

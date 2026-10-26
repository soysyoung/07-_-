"""실습 03 기본필수: 작은 정상 자료로 확률과 클래스를 확인합니다."""
from sklearn.linear_model import LogisticRegression

# 원리 확인용 데이터입니다. X의 한 값은 작업 복잡도, y는 성공 0 / 실패 1입니다.
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]
model = LogisticRegression()
# TODO B03: model을 X와 y로 학습하세요.
# 힌트: model.fit(입력, 정답)
pass
# fit(입력, 정답)은 주어진 사례를 이용해 분류 규칙을 학습하는 명령입니다.
# X에는 작업 복잡도가 들어 있고, 같은 위치의 y에는 실제 결과(성공 0, 실패 1)가 들어 있습니다.
# 예를 들어 X의 첫 번째 값 [1]에 대응하는 정답은 y의 첫 번째 값 0(성공)입니다.
# 로지스틱 회귀 모델은 이 자료를 보고 복잡도에 따른 성공·실패 확률의 변화를 학습합니다.
model.fit(X, y)

new = [[2], [5]]
print("클래스 순서:", model.classes_)
print("확률 [성공, 실패]:")
print(model.predict_proba(new).round(3))
print("예측 클래스:", model.predict(new))
# 이 작은 시연의 출력은 평가 점수가 아닙니다. 실제 성능 비교는 07번의 valid로 합니다.

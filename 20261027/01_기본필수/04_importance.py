"""실습 04: 정상 숫자 두 열의 중요도를 확인하는 작은 원리 예제입니다."""
from sklearn.ensemble import RandomForestClassifier

# 각 행은 [글자 수, 허용 도구 수]입니다. 별도 원리 예제이며 requests.csv와 다른 자료입니다.
X = [[100, 1], [200, 2], [300, 1], [700, 2], [800, 1], [900, 2]]
y = [0, 0, 0, 1, 1, 1]
names = ["글자 수", "허용 도구 수"]
model = RandomForestClassifier(n_estimators=20, max_depth=2, random_state=42, n_jobs=1)
# 숲을 먼저 학습해야 각 입력 정보가 예측에 얼마나 쓰였는지 계산할 수 있습니다.
model.fit(X, y)
# TODO B04: 학습한 model의 feature_importances_를 읽으세요.
# importance = None
# 학습된 모델에서 두 입력 항목의 중요도 점수를 가져옵니다.
# fit() 후: feature_importances_에 계산 결과가 저장됨
importance = model.feature_importances_
# TODO 끝
# 이름과 중요도 점수를 같은 순서끼리 짝지어 하나씩 출력합니다.
for name, value in zip(names, importance):
    print(f"{name}: {value:.3f}")
print("중요도가 높아도 실패의 원인이라고 단정하지 않습니다.")

"""실습 01: 실제 실패 6건인 손계산 예제입니다. 아직 모델을 학습하지 않습니다."""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# 행은 실제, 열은 예측입니다. 1은 좋은 결과가 아니라 관심 클래스인 '실패'입니다.
y = [0]*14 + [1]*6
pred = [0]*10 + [1]*4 + [0]*4 + [1]*2
print("혼동행렬(행 실제 / 열 예측, 성공0·실패1)")
print(confusion_matrix(y, pred, labels=[0, 1]))
# TODO B01: 네 지표 함수에 실제값 y와 예측값 pred를 순서대로 넣으세요. zero_division=0은 분모가 0일 때 0으로 표시하는 정책입니다.
# acc = precision = recall = f1 = 0.0

# 아래에서는 정답을 지표별로 계산합니다.
acc = accuracy_score(y, pred)
# TP(참): 실제 실패(1)를 실패(1)라고 맞힌 경우입니다.
# FP(거짓): 실제 성공(0)을 실패(1)라고 잘못 예측한 경우입니다.
# FN(거짓): 실제 실패(1)을 성공(0)이라고 놓친 경우입니다.
# TN(참): 실제 성공(0)을 성공(0)이라고 맞힌 경우입니다.

# TP (True Positive): 실제 양성을 양성으로 올바르게 예측함
# FP (False Positive): 실제 음성인데 양성으로 잘못 예측함 (1종 오류)
# FN (False Negative): 실제 양성인데 음성으로 잘못 예측함 (2종 오류)
# TN (True Negative): 실제 음성을 음성으로 올바르게 예측함

# 정밀도 = TP / (TP + FP): 실패라고 예측한 경우 중 실제 실패인 비율입니다.
precision = precision_score(y, pred, zero_division=0)
# 재현율 = TP / (TP + FN): 실제 실패 중 찾아낸 비율입니다.
recall = recall_score(y, pred, zero_division=0)
# F1: 정밀도와 재현율을 함께 보는 점수입니다.
f1 = f1_score(y, pred, zero_division=0)
# TODO 끝

print(f"Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
print("실제 실패 6건 중 발견: 2건")

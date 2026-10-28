"""실습 01 응용: 모델 예시와 전부 성공으로 예측하는 기준선을 비교합니다."""
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
y = [0]*14 + [1]*6
predictions = {"example": [0]*10 + [1]*4 + [0]*4 + [1]*2, "all_success": [0]*20}
rows = []
for name, pred in predictions.items():
    # ravel(): 배열을 1차원으로 펼치는 함수
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # TODO A01: 현재 후보의 F1을 계산하세요. 분모 0은 0으로 표시합니다.
    # f1 = 0.0

    # 실패(1, 여기서는 양성이라고 이해하자)를 찾아야 할 대상으로 봅니다.
    # TP(참양성): 실제 실패(양성)를 실패(양성)라고 맞힌 수입니다.
    # FP(거짓양성): 실제 성공인데 실패(양성)라고 잘못 예측한 수입니다.
    # FN(거짓음성): 실제 실패인데 성공(음성)이라고 놓친 수입니다.
    # TN(참음성): 실제 성공을 성공(음성)이라고 맞힌 수입니다.
    # 정밀도 = TP / (TP + FP), 재현율 = TP / (TP + FN)입니다.
    # F1 = 2 * TP / (2 * TP + FP + FN)입니다.
    # 분모가 0이면 나눌 수 없으므로 0으로 정합니다.
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    # TODO 끝
    rows.append({"model": name, "Accuracy": accuracy_score(y, pred),
                 "Precision": precision_score(y, pred, zero_division=0),
                 "Recall": recall_score(y, pred, zero_division=0), "F1": f1,
                 "FP": fp, "FN": fn, "TP": tp})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("정확도가 높은 all_success도 실패 6건을 하나도 찾지 못합니다.")

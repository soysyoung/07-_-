"""04 기본필수: 결측이 없는 숫자를 train 기준으로 표준화합니다."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 제공: 정상 숫자만 사용합니다. 결측 대체는 선택 단계에서 배웁니다.
train = pd.DataFrame({"input_chars": [100, 200, 300]})
valid = pd.DataFrame({"input_chars": [400, 500]})
scaler = StandardScaler()

# train 데이터로 평균과 표준편차를 학습합니다.
scaler.fit(train)

# transform은 이미 배운 평균과 크기를 사용합니다. 다시 fit하지 않습니다.
# TODO B04-1: scaler의 transform으로 valid를 변환하세요.
# train에서 학습한 평균과 표준편차를 사용하여 valid를 표준화합니다.
# valid에서는 fit을 다시 하지 않습니다.
valid_ready = scaler.transform(valid)

print("train에서 배운 평균:", scaler.mean_.tolist())
print("valid 변환 결과:")
print(valid_ready.round(3))

"""06 기본필수: 숫자 1열과 문자 1열을 함께 변환합니다."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

train = pd.DataFrame({"input_chars": [100, 200, 300],
                      "task_type": ["qa", "summary", "qa"]})
valid = pd.DataFrame({"input_chars": [400, 500],
                      "task_type": ["summary", "qa"]})
# 제공: 숫자에는 표준화, 문자에는 원-핫 변환을 지정합니다.
# ColumnTransformer: 열마다 서로 다른 전처리를 적용한 뒤 결과를 하나로 합치는 도구
# ColumnTransformer의 각 항목은 ("단계 이름", 변환기, 적용할 열)로 구성됩니다.
# "num"은 숫자형 열을 처리하는 단계에 붙인 이름입니다.
# "cat"은 범주형 열을 처리하는 단계에 붙인 이름입니다.
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])
preprocessor.fit(train)
# TODO B06-1: preprocessor의 transform으로 valid를 변환하세요.
valid_ready = preprocessor.transform(valid)

print("변환 전:", valid.shape)
print("변환 후:", valid_ready.shape)
print(valid_ready.round(3))

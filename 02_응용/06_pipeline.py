"""06 응용: 원래 PPT처럼 입력 4열을 숫자 8열로 변환합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 제공: data/는 입력 자료, outputs/는 실행 결과를 보관합니다. 이 파일의 위치를 기준으로 읽습니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "data" / "requests.csv")
# 제공: 먼저 20%를 test로 보관하고, 남은 80% 중 25%를 valid로 나눕니다.
# random_state는 재현용 고정값입니다. stratify는 성공/실패 비율을 고려합니다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])
# test는 개수 확인 외에 사용하지 않습니다.

# 제공: 숫자 2열은 중앙값 대체 후 표준화합니다.
# 숫자형 열의 전처리 과정을 하나의 Pipeline으로 묶습니다.
# 1. 결측값을 train에서 학습한 중앙값으로 대체합니다.
# 2. 대체가 끝난 값을 train의 평균과 표준편차로 표준화합니다.
numeric = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# 제공: 문자 2열은 최빈값 대체 후 0/1 열로 바꿉니다.
# 범주형 열의 전처리 과정을 하나의 Pipeline으로 묶습니다.
# train에서 보지 못한 범주는 오류 없이 모든 원-핫 열을 0으로 처리합니다.
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
# 열의 종류에 따라 서로 다른 Pipeline을 적용하고 결과를 하나로 합칩니다.
# "num"과 "cat"은 각 전처리 작업에 개발자가 붙인 이름입니다.
preprocessor = ColumnTransformer([
    ("num", numeric, ["input_chars", "allowed_tools"]),
    ("cat", categorical, ["task_type", "priority"]),
])
# 정답 failed와 식별자 request_id를 제외하고, 예측에 사용할 입력 열만 선택합니다.
features = ["input_chars", "allowed_tools", "task_type", "priority"]

# train에서 결측 대체값, 표준화 기준, 범주와 원-핫 열 순서를 학습하고 변환합니다.
train_ready = preprocessor.fit_transform(train[features])
# TODO A06-1: preprocessor로 valid를 변환하세요. train에서 다시 학습하지는 않습니다.
valid_ready = preprocessor.transform(valid[features])

print("train/valid 모양:", train_ready.shape, valid_ready.shape)
print("열 이름:", preprocessor.get_feature_names_out().tolist())

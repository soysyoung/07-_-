"""06 심화(기존 응용): 기준선 지표와 새 요청 예측을 연결합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

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
numeric = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
# 제공: 문자 2열은 최빈값 대체 후 0/1 열로 바꿉니다.
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric, ["input_chars", "allowed_tools"]),
    ("cat", categorical, ["task_type", "priority"]),
])
features = ["input_chars", "allowed_tools", "task_type", "priority"]

# 전처리와 분류 모델을 하나의 Pipeline으로 연결합니다.
# pre: 입력 열의 결측값 처리, 표준화, 원-핫 인코딩을 수행합니다.
# model: train에서 가장 자주 등장한 정답만 예측하는 기준선 모델입니다.
pipeline = Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])

# train 데이터로 전처리 기준과 분류 모델을 학습합니다.
pipeline.fit(train[features], train["failed"])
# TODO H06-1: valid의 입력 4열을 예측하세요.
# prediction = [1] * len(valid)
# 학습된 전처리 기준과 모델을 사용하여 valid의 실패 여부를 예측합니다.
prediction = pipeline.predict(valid[features])

# valid의 실제 정답과 예측값이 일치한 비율을 계산합니다.
accuracy = (prediction == valid["failed"].to_numpy()).mean()

# 실제 실패인 데이터 중 모델도 실패라고 예측한 개수를 계산합니다.
found = ((prediction == 1) & (valid["failed"].to_numpy() == 1)).sum()
# 아직 모델이 보지 못한 새로운 요청 2건을 만듭니다.
new_request = pd.DataFrame({"input_chars": [900, 1500], "allowed_tools": [2, 3],
                            "task_type": ["translation", "qa"],
                            "priority": ["urgent", "normal"]})
# TODO H06-2: 새 요청 2행 new_request를 예측하세요.
# new_prediction = pipeline.predict(valid[features].iloc[:2])
# train에서 학습한 전처리 기준과 모델로 새로운 요청의 실패 여부를 예측합니다.
new_prediction = pipeline.predict(new_request[features])

print("valid 정확도:", round(float(accuracy), 2))
print("실패 발견:", int(found), "/", int((valid["failed"] == 1).sum()))
print("새 요청 예측:", new_prediction.tolist())

"""실습 03 응용: PPT와 같은 100건에서 실패1의 예측을 확인합니다."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]
# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

model = Pipeline([("pre", pre), ("model", LogisticRegression(C=1.0, max_iter=1000))])
# TODO A03: 실패 여부로 학습하고 X_valid를 예측하세요.
# 힌트: 실패1의 정답 열은 failed입니다.
# pass
# prediction = None
# fit(문제, 정답)은 여러 사례를 보고 입력 특성과 실패 여부의 관계를 학습하는 과정입니다.
# X_train에는 입력 글자 수, 허용 도구 수, 작업 종류, 우선순위의 네 가지 특성이 들어 있습니다.
# train["failed"]에는 각 작업의 실제 결과가 들어 있으며, 0은 성공, 1은 실패를 뜻합니다.
# Pipeline이 전처리와 로지스틱 회귀를 순서대로 실행하므로 별도로 데이터를 변환할 필요가 없습니다.
model.fit(X_train, train["failed"])

# predict는 학습한 관계를 이용해 valid 데이터가 성공(0)할지 실패(1)할지 결정합니다.
# 예측 결과는 아래의 혼동행렬과 F1 점수를 계산할 때 사용됩니다.
prediction = model.predict(X_valid)

print("클래스 순서:", model.classes_)
print("앞 4건 실패 확률:", model.predict_proba(X_valid)[:4, 1].round(3))
print("혼동행렬 / 행=실제, 열=예측, 순서=[0,1]")
print(confusion_matrix(valid["failed"], prediction, labels=[0, 1]))
print("valid F1:", round(f1_score(valid["failed"], prediction, zero_division=0), 3))

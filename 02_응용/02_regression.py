"""실습 02 응용: 같은 valid 20건에서 두 회귀 모델을 비교합니다."""
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

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

rows = []
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression())]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A02: 회귀 정답으로 학습하고 같은 valid를 예측하세요.
    # 힌트: fit의 정답은 train["elapsed_ms"]입니다.
    # pass
    # prediction = None
    # fit(문제, 정답)은 여러 학습 사례를 보고 문제와 정답 사이의 규칙을 찾는 과정입니다.
    # 첫 번째 값인 train에는 입력 글자 수, 허용 도구 수, 작업 종류, 우선순위 등이 들어 있습니다.
    # 두 번째 값인 train["elapsed_ms"]에는 각 작업에 실제로 걸린 시간(ms), 즉 정답이 들어 있습니다.
    # 전처리기(pre)가 features에 지정된 네 열만 골라 사용하므로 elapsed_ms가 문제에 섞이지 않습니다.
    model.fit(train, train["elapsed_ms"])

    # predict(새로운 문제)는 학습한 규칙을 이용해 아직 정답을 모른다고 가정한 데이터의 값을 예측합니다.
    # 여기서는 valid의 각 작업에 걸릴 시간을 예측하고, 그 결과를 prediction에 저장합니다.
    prediction = model.predict(valid)

    rows.append({"model": name, "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
                 "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})
comparison = pd.DataFrame(rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

comparison.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("저장: outputs/regression_comparison.csv")

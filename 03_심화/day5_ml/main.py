"""실습 07: 전처리 구성·실행·결과·보고서의 위치를 나눈 작은 프로젝트입니다."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import build_pipeline

# 제공: 경로 처리는 학습 과제가 아닙니다. 다른 작업 폴더에서 실행해도 이 파일을 기준으로 읽습니다.
HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
data = pd.read_csv(HERE / "data" / "requests.csv")

# 제공: 원본 100행의 순서와 두 seed를 유지하여 전날과 같은 60/20/20을 만듭니다.
# 전체 데이터의 20%를 최종 test 데이터로 따로 보관합니다.
# failed의 성공·실패 비율이 각 데이터에 비슷하게 유지되도록 나눕니다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])

# 남은 데이터의 25%를 valid로 분리합니다.
# 최종 비율은 train 60%, valid 20%, test 20%가 됩니다.
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])

# ID와 정답을 제외하고 모델이 예측에 사용할 입력 열만 선택합니다.
features = ["input_chars", "allowed_tools", "task_type", "priority"]

# preprocessing.py는 처리 순서만 구성합니다. 실제 학습은 train으로만 여기에서 수행합니다.
# preprocessing.py에 정의된 전처리기와 기준선 모델을 가져옵니다.
pipeline = build_pipeline()

# train 데이터만 사용하여 전처리 기준과 모델을 학습합니다.
pipeline.fit(train[features], train["failed"])
# TODO H07-1: 학습된 pipeline으로 valid의 입력 4열을 예측하세요.
# prediction = [1] * len(valid)
prediction = pipeline.predict(valid[features])

# 제공: 각 예측을 원래 요청 ID와 연결합니다. 원본 requests.csv는 변경하지 않습니다.
# 요청 ID, 실제 정답, 예측 결과를 하나의 표로 만듭니다.
result = valid[["request_id", "failed"]].copy()
result["predicted"] = prediction

# 나중에 확인할 수 있도록 valid 예측 결과를 CSV 파일로 저장합니다.
result.to_csv(OUTPUTS / "validation_predictions.csv", index=False, encoding="utf-8-sig")

# 제공: 세 묶음의 ID와 묶음 안의 순서(position)를 남깁니다. test의 특징·정답은 모델에 전달하지 않습니다.
# 각 요청이 train, valid, test 중 어디에 들어갔는지 기록합니다.
# position에는 해당 데이터 묶음 안에서의 행 순서를 저장합니다.
manifest_parts = []
for split_name, part in [("train", train), ("valid", valid), ("test", test)]:
    manifest_parts.append(pd.DataFrame({
        "request_id": part["request_id"].tolist(),
        "split": split_name,
        "position": range(len(part)),
    }))

# 따로 만든 train, valid, test 기록을 하나의 표로 합칩니다.
manifest = pd.concat(manifest_parts, ignore_index=True)

# 같은 데이터를 다시 나누지 않아도 되도록 분할 결과를 저장합니다.
manifest.to_csv(OUTPUTS / "split_manifest.csv", index=False, encoding="utf-8-sig")

# 제공: 학습된 열 이름과 전처리+모델을 저장합니다. 다른 사람이 보낸 joblib 파일은 열지 않습니다.
# 전처리 후 만들어진 숫자 열의 이름을 확인합니다.
feature_names = pipeline["pre"].get_feature_names_out().tolist()

# 변환된 열 이름을 JSON 파일로 저장합니다.
(OUTPUTS / "feature_names.json").write_text(
    json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")

# 학습된 전처리기와 모델을 하나의 파일로 저장합니다.
# restore.py에서 이 파일을 다시 불러와 같은 예측이 나오는지 확인합니다.
joblib.dump(pipeline, OUTPUTS / "pipeline.joblib")

# 실제 정답과 예측값이 같은 행의 비율을 계산합니다.
accuracy = (result["failed"] == result["predicted"]).mean()

# 실제 실패이면서 모델도 실패라고 맞힌 요청의 개수를 계산합니다.
found = ((result["failed"] == 1) & (result["predicted"] == 1)).sum()
print("분할:", len(train), len(valid), len(test))
print("변환 특성 수:", len(feature_names))
print("valid 정확도:", round(accuracy, 2))
print("실패 발견:", int(found), "/", int((result["failed"] == 1).sum()))
print("저장: outputs/validation_predictions.csv")
print("저장: outputs/split_manifest.csv, outputs/feature_names.json")
print("저장: outputs/pipeline.joblib")
print("08교시: restore.py 실행 후 reports/model_review.md를 작성합니다.")

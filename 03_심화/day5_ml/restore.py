"""실습 08: 먼저 같은 폴더의 main.py를 완성하고 실행한 뒤, 이 파일을 별도로 실행합니다."""
from pathlib import Path
import joblib
import pandas as pd

# main.py에서 저장한 원본 데이터와 결과 파일의 위치를 지정합니다.
HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

# valid 데이터를 다시 만들기 위해 원본 요청 데이터를 읽습니다.
data = pd.read_csv(HERE / "data" / "requests.csv")

# 제공: 저장한 분할표를 읽어 같은 ID를 같은 순서로 꺼냅니다. 새로 분할하거나 학습하지 않습니다.
# main.py에서 저장한 데이터 분할 기록을 읽습니다.
manifest = pd.read_csv(OUTPUTS / "split_manifest.csv")

# valid에 속했던 요청 ID를 당시의 순서대로 가져옵니다.
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()

# 원본 데이터에서 같은 요청 ID를 같은 순서로 꺼내 valid를 복원합니다.
valid = data.set_index("request_id").loc[valid_ids].reset_index()

# 모델이 예측에 사용할 입력 열을 지정합니다.
features = ["input_chars", "allowed_tools", "task_type", "priority"]

# 직접 실행한 main.py가 같은 환경에서 만든 파일만 복원합니다.
# main.py에서 학습하고 저장한 Pipeline을 다시 불러옵니다.
restored = joblib.load(OUTPUTS / "pipeline.joblib")

# main.py가 저장했던 valid 예측 결과를 읽습니다.
saved = pd.read_csv(OUTPUTS / "validation_predictions.csv")
# TODO H08-1: 복원한 restored로 valid 입력을 예측하세요. fit은 호출하지 마세요.
# prediction = saved["predicted"].to_numpy() + 1

# 복원한 Pipeline으로 같은 valid 입력을 다시 예측합니다.
# 이미 학습된 모델이므로 fit은 다시 호출하지 않습니다.
prediction = restored.predict(valid[features])

# 제공: 예측 숫자뿐 아니라 요청 ID의 순서도 함께 대조합니다.
# 복원한 valid의 요청 ID와 저장된 예측 파일의 요청 ID 순서가 같은지 확인합니다.
same_ids = valid["request_id"].tolist() == saved["request_id"].tolist()

# 복원한 모델의 예측과 main.py에서 저장한 예측이 모두 같은지 확인합니다.
same = (prediction == saved["predicted"].to_numpy()).all()

print("요청 ID 순서 동일:", same_ids)
print("저장 전후 예측 동일:", bool(same))
print("valid 예측 건수:", len(prediction))
print("복원한 모델에 fit을 다시 호출하지 않았습니다.")

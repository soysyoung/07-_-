"""제공 모듈: 전처리와 기준선의 구성만 담당합니다. 이 파일은 수정하지 않습니다."""

# 숫자나 문자 데이터의 결측값을 평균·중앙값·최빈값 등으로 채우는 도구입니다.
from sklearn.impute import SimpleImputer
# StandardScaler는 숫자형 데이터를 표준화하고,
# OneHotEncoder는 범주형 문자열을 0과 1의 숫자 열로 변환합니다.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 열의 종류에 따라 서로 다른 전처리를 적용한 뒤 결과를 하나로 합치는 도구입니다.
from sklearn.compose import ColumnTransformer
# 여러 전처리 단계와 모델을 정해진 순서로 연결하는 도구입니다.
from sklearn.pipeline import Pipeline
# 가장 자주 등장한 정답 등을 반복해서 예측하는 단순한 기준선 모델입니다.
# 이후 실제 모델의 성능이 이 기준보다 좋은지 비교할 때 사용합니다.
from sklearn.dummy import DummyClassifier


def build_pipeline():
    """아직 학습하지 않은 새 Pipeline을 돌려줍니다. fit은 실행 파일에서 호출합니다."""
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])

    # 범주형 열은 결측값을 최빈값으로 채운 후 원-핫 인코딩합니다.
    # 처음 보는 범주가 들어와도 오류가 발생하지 않도록 설정합니다.
    categorical = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # 숫자형 열과 범주형 열에 서로 다른 전처리를 적용한 뒤 합칩니다.
    preprocessor = ColumnTransformer([
        ("num", numeric, ["input_chars", "allowed_tools"]),
        ("cat", categorical, ["task_type", "priority"]),
    ])

    # 전처리기와 가장 흔한 정답만 예측하는 기준선 모델을 연결합니다.
    # 아직 학습하지 않은 Pipeline이며, main.py에서 fit을 실행합니다.
    return Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])

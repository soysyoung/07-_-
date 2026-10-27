"""Day7 실습. 실행: python main.py. 결과: 이 폴더의 outputs/."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import f1_score, log_loss
from sklearn.inspection import permutation_importance
from course_support import (load_data, build, make_pre, snapshot, emit, write_json,
    cls_metrics, reg_metrics, check_features, NUM, CAT, FEATURES)
from model_config import new_classifier, new_regressor, make_folds, fold_table

def importance_bias_demo(seed=42):
    """공통100건을 변경하지 않는 별도 합성320건 실험. raw4 대신 3개 수치열.
    반환: 학습 MDI / holdout 순열 F1감소 표. 순위 방향을 강제로 만들지 않는다.
    """
    # TODO D7-04-H1: 위 함수 설명과 README 계약에 맞게 구현하세요.
    # raise NotImplementedError("TODO D7-04-H1")
    # 같은 seed를 사용하면 실행할 때마다 똑같은 가상 자료가 만들어집니다.
    rng = np.random.default_rng(seed)
    count = 320
    signal = rng.normal(size=count)
    X = pd.DataFrame({
        "signal": signal,
        # 원래 신호에 작은 흔들림만 더해 서로 매우 비슷한 복제 열을 만듭니다.
        "signal_copy": signal + rng.normal(scale=0.2, size=count),
        # 정답과 관계없이 따로 만든 숫자로, 값의 종류가 많은 잡음 열입니다.
        "independent_high_cardinality_noise": rng.uniform(size=count),
    })
    # 신호에 잡음을 섞어 너무 쉽게 정답을 맞히지 못하도록 0과 1을 만듭니다.
    y = (signal + rng.normal(scale=1.2, size=count) > 0).astype(int)

    # 앞의 240건으로 학습하고, 뒤의 80건은 중요도를 확인할 때만 사용합니다.
    fit_slice = slice(0, 240)
    holdout_slice = slice(240, count)
    forest = clone(new_classifier("Forest"))
    forest.fit(X.iloc[fit_slice], y[fit_slice])

    # holdout의 각 열을 여러 번 섞어 F1 점수가 얼마나 떨어지는지 확인합니다.
    shuffled = permutation_importance(
        forest, X.iloc[holdout_slice], y[holdout_slice],
        scoring="f1", n_repeats=20, random_state=seed, n_jobs=1,
    )
    return pd.DataFrame({
        "feature": X.columns,
        "MDI": forest.feature_importances_,
        "F1_drop_mean": shuffled.importances_mean,
        "repeat_std": shuffled.importances_std,
    })

def run():
    return {'separate_bias_experiment':importance_bias_demo(),
        'not_main_dataset':True,
        'warning':'복제열은 중요도를 나눠 가질 수 있다. 잡음열의 특정 순위를 보장하지 않는다. 이 결과로 공통 데이터의 열을 삭제하지 않는다.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'outputs')
    args = parser.parse_args()
    emit(run(), args.output)

if __name__ == '__main__':
    main()

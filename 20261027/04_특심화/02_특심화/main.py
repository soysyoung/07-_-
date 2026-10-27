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

def diagnose(d):
    """사전 3후보만 비교. case, depth, leaf, train_F1, valid_F1, gap 표."""
    # TODO D7-02-H1: 위 함수 설명과 README 계약에 맞게 구현하세요.
    # raise NotImplementedError("TODO D7-02-H1")
    # 얕은 나무, 기준 나무, 제한 없는 나무를 같은 자료로 비교합니다.
    settings = [("restricted", 1, 5), ("reference", 3, 3),
                ("unrestricted", None, 1)]
    rows = []
    for case, depth, leaf in settings:
        model = build(DecisionTreeClassifier(max_depth=depth,
                      min_samples_leaf=leaf, random_state=42))
        model.fit(d.X_train, d.yc_train)
        train_f1 = f1_score(d.yc_train, model.predict(d.X_train), zero_division=0)
        valid_f1 = f1_score(d.yc_valid, model.predict(d.X_valid), zero_division=0)
        rows.append({"case": case, "depth": str(depth), "leaf": leaf,
                     "train_F1": train_f1, "valid_F1": valid_f1,
                     "gap": train_f1 - valid_f1})
    return pd.DataFrame(rows)

def run():
    return {'diagnostics':diagnose(load_data()),'test_evaluated':False,
        'warning':'valid20건 진단만으로 정상/과소/과적합을 확정하지 않는다. 최종 기본 후보는 변경하지 않는다.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'outputs')
    args = parser.parse_args()
    emit(run(), args.output)

if __name__ == '__main__':
    main()

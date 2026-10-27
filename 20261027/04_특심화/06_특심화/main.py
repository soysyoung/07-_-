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

def audit_medians(pipe, X, y, folds):
    """각 fold fit된 중앙값과 X.iloc[fit][NUM].median을 대조한 표 반환."""
    # TODO D7-06-H1: 위 함수 설명과 README 계약에 맞게 구현하세요.
    # raise NotImplementedError("TODO D7-06-H1")
    # Pipeline 전체를 교차검증해 각 fold마다 전처리기도 새로 학습합니다.
    checked = cross_validate(pipe, X, y, cv=folds, scoring="f1",
                             return_estimator=True, n_jobs=1, error_score="raise")
    rows = []
    for number, ((fit_positions, _), fitted) in enumerate(
            zip(folds, checked["estimator"]), start=1):
        actual = fitted["pre"].named_transformers_["num"]["imputer"].statistics_
        expected = X.iloc[fit_positions][NUM].median().to_numpy()
        for feature, learned, fit_only in zip(NUM, actual, expected):
            rows.append({"fold": number, "feature": feature, "actual": learned,
                         "expected_fit_only": fit_only,
                         "passed": bool(np.isclose(learned, fit_only))})
    return pd.DataFrame(rows)

def search_four(d, folds):
    """depth[2,4]×leaf[1,3]만 탐색. 학습된 GridSearchCV 반환."""
    # TODO D7-06-H2: 위 함수 설명과 README 계약에 맞게 구현하세요.
    # raise NotImplementedError("TODO D7-06-H2")
    # 두 깊이와 두 잎 크기를 조합하여 모두 네 후보만 비교합니다.
    grid = GridSearchCV(
        build(new_classifier("Tree")),
        {"model__max_depth": [2, 4], "model__min_samples_leaf": [1, 3]},
        scoring="f1", cv=folds, refit=True, n_jobs=1, error_score="raise",
    )
    return grid.fit(d.X_train, d.yc_train)

def run():
    d=load_data(); folds=make_folds(d.X_train,d.yc_train)
    audit=audit_medians(build(new_classifier('Tree')),d.X_train,d.yc_train,folds)
    grid=search_four(d,folds)
    rows=pd.DataFrame(grid.cv_results_)
    columns=['param_model__max_depth','param_model__min_samples_leaf','mean_test_score','std_test_score','rank_test_score']
    return {'fold_statistics':audit,'grid_results':rows[columns],
        'best_params':grid.best_params_,'best_CV_F1':float(grid.best_score_),
        'refit_n':len(d.X_train),'experiment_name':'TreeGrid (별도 탐색, 기본Tree 교체 아님)',
        'test_evaluated':False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'outputs')
    args = parser.parse_args()
    emit(run(), args.output)

if __name__ == '__main__':
    main()

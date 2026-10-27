"""선택 심화: train60 안에서 inner42/stop18로 조기 종료를 결정합니다.
PPT82~84, E11/E12/E21. 기본 XGB80 비교와 분리된 XGB_ES 실험입니다.
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sklearn.model_selection import train_test_split
from course_support import load_data,make_pre,build,emit,cls_metrics
from model_config import new_classifier

def inner_preprocess(d):
    """train60 -> inner42/stop18, stratify+seed42. pre는 inner42에서만 fit.
    반환: pre, X_inner, X_stop, y_inner, y_stop, Z_inner, Z_stop.
    """
    # TODO D7-BONUS-ES1: 함수 설명에 따라 작성하세요.
    # raise NotImplementedError("TODO D7-BONUS-ES1")
    # train 60건만 42건과 18건으로 나누며 두 모둠의 클래스 비율을 맞춥니다.
    X_inner, X_stop, y_inner, y_stop = train_test_split(
        d.X_train, d.yc_train, test_size=0.3,
        stratify=d.yc_train, random_state=42)
    # 전처리 통계는 inner 42건에서만 배우고 stop에는 그대로 적용합니다.
    pre = make_pre()
    Z_inner = pre.fit_transform(X_inner)
    Z_stop = pre.transform(X_stop)
    return pre, X_inner, X_stop, y_inner, y_stop, Z_inner, Z_stop

def fit_early_stop(Zi,yi,Zs,ys):
    """최대300회차/patience15, 마지막 eval_set=stop18, 진단logloss. fit된 모델 반환."""
    # TODO D7-BONUS-ES2: 함수 설명에 따라 작성하세요.
    # raise NotImplementedError("TODO D7-BONUS-ES2")
    # stop 점수가 15회 연속 좋아지지 않으면 학습을 일찍 멈춥니다.
    model = new_classifier("XGB").set_params(
        n_estimators=300, early_stopping_rounds=15)
    model.fit(Zi, yi, eval_set=[(Zi, yi), (Zs, ys)], verbose=False)
    return model

def run():
    d=load_data();pre,Xi,Xs,yi,ys,Zi,Zs=inner_preprocess(d)
    model=fit_early_stop(Zi,yi,Zs,ys)
    rounds=model.best_iteration+1  # best_iteration은 0부터 세므로 +1
    # 새 전처리기 + 조기종료 없는 새 모델을 train60 전체로 다시 학습합니다.
    refit=build(new_classifier('XGB').set_params(n_estimators=rounds)).fit(d.X_train,d.yc_train)
    history=model.evals_result()
    curve=pd.DataFrame({'round':np.arange(1,len(history['validation_1']['logloss'])+1),
        'inner_logloss':history['validation_0']['logloss'],
        'stop_logloss':history['validation_1']['logloss']})
    return {'experiment':'XGB_ES (기본 XGB80과 별도)','inner_n':len(Xi),'stop_n':len(Xs),
        'overlap':len(set(Xi.index)&set(Xs.index)),
        'inner_numeric_medians':pre.named_transformers_['num']['imputer'].statistics_,
        'best_iteration':model.best_iteration,'selected_rounds':rounds,
        'executed_rounds':len(curve),'stop_logloss':float(model.best_score),
        'refit_train_n':len(d.X_train),'refit_early_stopping':refit['model'].get_params()['early_stopping_rounds'],
        'refit_valid_metrics':cls_metrics(d.yc_valid,refit.predict(d.X_valid)),
        'learning_curve':curve,'inner_ids':list(Xi.index),'stop_ids':list(Xs.index),
        'test_evaluated':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs')
    args=p.parse_args();emit(run(),args.output)

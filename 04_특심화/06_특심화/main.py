"""가중치 추가 실험과 clone | PPT 61 · 86쪽 | 심화. main.py의 핵심 함수를 완성합니다."""
from pathlib import Path
import sys
# 터미널의 현재 폴더와 무관하게 이 실습의 배포 루트를 찾습니다.
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'course_support.py').is_file())
sys.path.insert(0,str(ROOT))
import argparse
import numpy as np
import pandas as pd
from course_support import (NUM,CAT,FEATURES,DATA,load_data,load_checkpoint,
    make_pre,build,reg_metrics,cls_metrics,snapshot,emit,read_table)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

def compare_weights(d):
    """분류 K=3·standard·p=2를 고정, weights 두 후보만 비교.
    반환 표: weights, train_F1, valid_F1. 동점이면 uniform을 우선한다.
    """
    rows=[]
    for weights in ['uniform','distance']:
        # 두 후보는 가중치만 다르고 K, 거리, 전처리, 데이터는 모두 같습니다.
        estimator=KNeighborsClassifier(n_neighbors=3,weights=weights,p=2)
        model=build(estimator,scale='standard').fit(d.X_train,d.yc_train)
        rows.append({'weights':weights,
                     'train_F1':cls_metrics(d.yc_train,model.predict(d.X_train))['F1'],
                     'valid_F1':cls_metrics(d.yc_valid,model.predict(d.X_valid))['F1']})
    return pd.DataFrame(rows)

def clone_check(d):
    """base K5는 fit하지 않고 복제본 K3만 학습. base_K,trial_K,base_unfitted를 dict로 반환."""
    base=KNeighborsClassifier(n_neighbors=5)
    trial=clone(base)
    trial.set_params(n_neighbors=3)
    # 이 확인에서는 전처리 성능을 비교하지 않으므로 한 열의 결측값만 train 중앙값으로 채웁니다.
    # base는 그대로 두고 trial만 fit해야 두 객체의 학습 상태가 분리되었는지 확인할 수 있습니다.
    one_feature=d.X_train[['input_chars']].fillna(d.X_train['input_chars'].median())
    trial.fit(one_feature,d.yc_train)
    # clone은 설정을 복사하지만 학습 결과(classes_ 등)는 복사하지 않습니다.
    return {'base_K':base.n_neighbors,
            'trial_K':trial.n_neighbors,
            'base_unfitted':not hasattr(base,'classes_')}

def solve():
    d=load_data(); table=compare_weights(d)
    ranked=table.assign(tie_rank=table.weights.map({'uniform':0,'distance':1}))
    best=ranked.sort_values(['valid_F1','tie_rank'],ascending=[False,True]).iloc[0]['weights']
    return {'weights_comparison':table,'selected_weights':best,'clone_check':clone_check(d),
            'scope':'가중치만 추가한 2후보 보조 실험. test를 보지 않으며 모든 조합의 최적값 주장이 아님.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs')
    args=parser.parse_args()
    # 교육 환경의 CPU 수에 따른 과도한 병렬화를 줄입니다. 계산 정의는 바뀌지 않습니다.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result=solve()
    emit(result,args.output)

if __name__=='__main__':
    main()

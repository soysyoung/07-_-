"""거리 선택·동률·K 범위 | PPT 39~42 · 87쪽 | 심화. main.py의 핵심 함수를 완성합니다."""
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
import numbers

def guard_k(k,n_train):
    """bool/실수는 허용하지 않는다. 정수 1<=K<=n_train이면 K 반환, 아니면 ValueError."""
    # bool은 파이썬에서 정수처럼 보이지만 K로 쓰면 뜻이 불분명하므로 거부합니다.
    if isinstance(k,bool) or not isinstance(k,numbers.Integral) or not 1<=k<=n_train:
        raise ValueError(f'K는 1~{n_train}의 정수여야 합니다: {k!r}')
    return int(k)

def distance_choice(X,y,query):
    """metric=minkowski, K=1을 고정하고 p=1·2의 이웃 index·거리·예측 표 반환."""
    rows=[]
    for p in [1,2]:
        # p=1은 맨해튼 거리, p=2는 우리가 흔히 쓰는 직선거리입니다.
        model=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=p,
                                   algorithm='brute').fit(X,y)
        distance,index=model.kneighbors(query,n_neighbors=1)
        rows.append({'p':p,'neighbor_index':int(index[0,0]),
                     'distance':float(distance[0,0]),
                     'prediction':int(model.predict(query)[0])})
    return pd.DataFrame(rows)

def solve():
    X=np.array([[3,0],[2,2],[8,8]]); y=np.array([0,1,1])
    choices=distance_choice(X,y,np.array([[0,0]]))
    # 같은 거리의 이웃이 K 경계에서 경쟁할 때의 행 순서 영향을 관찰합니다.
    tie=[]
    for order in [[0,1],[1,0]]:
        xt=np.array([[-1],[1]])[order]; yt=np.array([0,1])[order]
        m=KNeighborsClassifier(n_neighbors=1,algorithm='brute').fit(xt,yt)
        tie.append({'order':order,'prediction':int(m.predict([[0]])[0])})
    caught=[]
    for k in [0,4,1.5,True]:
        try: guard_k(k,3)
        except ValueError: caught.append(str(k))
    m=KNeighborsClassifier(n_neighbors=3).fit([[1],[2]],[0,1])
    try: m.predict([[1.5]])
    except ValueError: error='fit은 성공했지만 predict에서 이웃 수가 부족하여 실패'
    return {'distance_choice':choices,'tie_order_observation':tie,'rejected_K':caught,
            'predict_time_failure':error,'tie_warning':'동률 결과를 모든 구현·버전에서 같은 순서라고 일반화하지 않음'}


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

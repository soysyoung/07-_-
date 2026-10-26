"""거리 기여와 전체 배열 표준화 | PPT 47~52쪽 | 심화. main.py의 핵심 함수를 완성합니다."""
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def distance_components(a,b,n_numeric=2):
    """변환된 두 행의 차이를 제곱해 수치/범주/전체 제곱거리 dict로 반환한다.
    key: numeric_squared, categorical_squared, total_squared.
    전체는 두 성분의 합이며 원래 유클리드 거리는 그 제곱근이다.
    """
    a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or not 0<=n_numeric<=len(a):
        raise ValueError('두 행의 모양과 수치 열 개수를 확인하세요.')
    squared=(a-b)**2
    numeric=float(squared[:n_numeric].sum())
    categorical=float(squared[n_numeric:].sum())
    return {'numeric_squared':numeric,
            'categorical_squared':categorical,
            'total_squared':numeric+categorical}

def compare_whole_scaling(d):
    """기존 수치-only와 전체 배열(원핫 포함) 표준화를 별도 실험명으로 비교한다.
    반환: experiment, valid_F1 표. 전체 배열 경로는 make_pre('none')→StandardScaler→KNN5.
    """
    rows=[]
    # 기존 방식: 숫자 열만 표준화하고 원핫 범주 열은 0과 1을 그대로 사용합니다.
    numeric_only=build(KNeighborsClassifier(n_neighbors=5),scale='standard')
    numeric_only.fit(d.X_train,d.yc_train)
    rows.append({'experiment':'numeric_only',
                 'valid_F1':cls_metrics(d.yc_valid,numeric_only.predict(d.X_valid))['F1']})
    # 별도 실험: 전처리가 만든 전체 8열을 다시 표준화합니다.
    whole=Pipeline([('pre',make_pre('none')),
                    ('whole_scaler',StandardScaler()),
                    ('model',KNeighborsClassifier(n_neighbors=5))])
    whole.fit(d.X_train,d.yc_train)
    rows.append({'experiment':'whole_including_onehot',
                 'valid_F1':cls_metrics(d.yc_valid,whole.predict(d.X_valid))['F1']})
    return pd.DataFrame(rows)

def solve():
    d=load_data(); pre=make_pre().fit(d.X_train)
    train_z=pre.transform(d.X_train); valid_z=pre.transform(d.X_valid)
    return {'distance_contribution':distance_components(train_z[0],valid_z[0]),
            'id_pair':[d.X_train.index[0],d.X_valid.index[0]],
            'alternative_scaling':compare_whole_scaling(d),
            'note':'범주 원핫의 거리 가중도 바뀌는 별도 실험. 기존 none/standard/minmax 비교와 섞지 않음'}


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

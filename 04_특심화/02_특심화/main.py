"""지표 선택과 train-only 이상치 정책 | PPT 18~23쪽 | 심화. main.py의 핵심 함수를 완성합니다."""
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
from sklearn.linear_model import LinearRegression

def iqr_mask(y_train):
    """train y의 Q1,Q3와 1.5*IQR로 경계를 정한다. (keep bool 배열, [lower,upper]) 반환.
    valid y를 인자로 받지 않으므로 valid 결과를 본 뒤 행을 지울 수 없다.
    """
    y=np.asarray(y_train)
    # Q1은 아래쪽 25%, Q3는 아래쪽 75% 지점입니다.
    q1,q3=np.percentile(y,[25,75])
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    # 경계 안에 있는 train 행만 True로 표시합니다.
    keep=(y>=lower)&(y<=upper)
    return keep,[float(lower),float(upper)]

def compare_policy(X_train,y_train,X_valid,y_valid):
    """원래 train/사전 IQR 정책 train으로 학습. 동일 valid에 대한 결과 표를 반환한다.
    model: all_train, train_iqr. n_train, MAE_ms, RMSE_ms, R2도 기록.
    """
    keep,bounds=iqr_mask(y_train)
    rows=[]
    for name,mask in [('all_train',np.ones(len(y_train),dtype=bool)),
                      ('train_iqr',keep)]:
        # 정책마다 새 모델을 만들어 같은 valid에 예측합니다.
        model=LinearRegression().fit(np.asarray(X_train)[mask],np.asarray(y_train)[mask])
        prediction=model.predict(X_valid)
        rows.append({'model':name,'n_train':int(mask.sum()),
                     **reg_metrics(y_valid,prediction)})
    return pd.DataFrame(rows),bounds

def solve():
    # 별도 원리 예제입니다. 공통 100건 데이터에서 이상치를 삭제하지 않습니다.
    X=np.arange(1,7).reshape(-1,1); y=np.array([100,200,300,400,500,5000])
    Xv=np.array([[7],[8],[9]]); yv=np.array([700,800,900])
    table,bounds=compare_policy(X,y,Xv,yv)
    custom={name:reg_metrics([100,200,300,400],p) for name,p in
            {'A':[100,200,300,560],'B':[150,250,350,450]}.items()}
    return {'policy_comparison':table,'train_iqr_bounds':bounds,'custom_metric_tradeoff':custom,
            'valid_kept':len(Xv),'note':'개선은 이 교육용 자료의 결과이며 이상치 자동삭제를 일반화하지 않습니다.'}


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

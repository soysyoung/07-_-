"""실습06의 K 실험을 종합 프로젝트로 옮깁니다. test는 함수 인자로도 받지 않습니다."""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from preprocessing import build_pipeline
from evaluate import regression_metrics,classification_metrics
from course_io import validate_k

def search_k(d,candidates=(1,3,5,9,15)):
    """task,K,scale,weights,p,metric,train_score,valid_score의 10행 표를 만든다.
    회귀/분류는 서로 다른 y와 지표를 사용한다. 조건은 standard,uniform,p=2로 고정한다.
    """
    rows=[]
    for task in ['classification','regression']:
        y_train=d.yc_train if task=='classification' else d.yr_train
        y_valid=d.yc_valid if task=='classification' else d.yr_valid
        for k in candidates:
            k=validate_k(k,len(d.X_train))
            estimator=(KNeighborsClassifier(n_neighbors=k,weights='uniform',p=2)
                       if task=='classification'
                       else KNeighborsRegressor(n_neighbors=k,weights='uniform',p=2))
            # K 하나마다 새 Pipeline을 학습해야 후보끼리 학습 상태가 섞이지 않습니다.
            model=build_pipeline(estimator).fit(d.X_train,y_train)
            train_prediction=model.predict(d.X_train)
            valid_prediction=model.predict(d.X_valid)
            if task=='classification':
                metric='F1'
                train_score=classification_metrics(y_train,train_prediction)['F1']
                valid_score=classification_metrics(y_valid,valid_prediction)['F1']
            else:
                metric='RMSE_ms'
                train_score=regression_metrics(y_train,train_prediction)['RMSE_ms']
                valid_score=regression_metrics(y_valid,valid_prediction)['RMSE_ms']
            rows.append({'task':task,'K':k,'scale':'standard','weights':'uniform',
                         'p':2,'metric':metric,
                         'train_score':train_score,'valid_score':valid_score})
    return pd.DataFrame(rows)

def choose_k(table):
    result={}
    for task in ['classification','regression']:
        p=table.loc[table.task==task].sort_values(['valid_score','K'],
                    ascending=[task=='regression',True],kind='stable')
        if p.empty: raise ValueError('K 탐색표에 두 문제의 결과가 모두 필요합니다.')
        result[task]=int(p.iloc[0]['K'])
    return result

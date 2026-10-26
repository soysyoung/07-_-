"""지표·학습·검증 담당. 비교표의 점수는 모두 validation 결과입니다."""
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,root_mean_squared_error,r2_score,
    accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
from preprocessing import build_pipeline
from checks import before_model,check_finite
from course_io import snapshot

def regression_metrics(y,pred):
    """실습02 재사용. MAE_ms, RMSE_ms, R2 dict를 계산한다. 원시 값을 반올림하지 않는다."""
    # 회귀는 숫자의 차이를 재므로 ms 단위 지표와 단위가 없는 R²를 함께 기록합니다.
    return {'MAE_ms':float(mean_absolute_error(y,pred)),
            'RMSE_ms':float(root_mean_squared_error(y,pred)),
            'R2':float(r2_score(y,pred))}

def classification_metrics(y,pred):
    """실습03 재사용. labels=[0,1]. Accuracy,Precision,Recall,F1,TN,FP,FN,TP 반환.
    실패=1, zero_division=0. 오탐은 FP, 실제 실패를 놓친 것은 FN이다.
    """
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'Accuracy':float(accuracy_score(y,pred)),
            'Precision':float(precision_score(y,pred,zero_division=0)),
            'Recall':float(recall_score(y,pred,zero_division=0)),
            'F1':float(f1_score(y,pred,zero_division=0)),
            'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def fit_candidates(candidates,X_train,y_train,X_valid,y_valid,task):
    """{이름:estimator}별로 독립 Pipeline을 학습한다.
    반환 (comparison DataFrame, long prediction DataFrame, fitted_models dict, trace list).
    task는 regression/classification. train-only fit, valid predict, 입력·통계 검사를 포함한다.
    make_records()는 제공 코드다. 학생은 핵심 학습 흐름을 구현한다.
    """
    if task not in {'regression','classification'}:
        raise ValueError('task는 regression 또는 classification이어야 합니다.')
    rows=[];records=[];models={};trace=[]
    for name,estimator in candidates.items():
        # 후보마다 입력과 K를 먼저 검사하고 독립된 Pipeline을 만듭니다.
        Xtr,Xva=before_model(estimator,X_train,y_train,X_valid,y_valid)
        model=build_pipeline(estimator)
        model.fit(Xtr,y_train)
        fitted_statistics=snapshot(model['pre'])
        prediction=model.predict(Xva)
        check_finite(prediction,len(Xva))
        unchanged=fitted_statistics==snapshot(model['pre'])
        if not unchanged:
            raise AssertionError('valid 예측 중 train 전처리 통계가 바뀌었습니다.')

        fitted=model['model']
        common={'task':task,'model':name,'scale':'standard',
                'n_train':len(Xtr),'n_valid':len(Xva),
                'K':getattr(fitted,'n_neighbors',np.nan),
                'C':getattr(fitted,'C',np.nan),
                'weights':getattr(fitted,'weights','not_applicable'),
                'p':getattr(fitted,'p',np.nan)}
        if task=='regression':
            train_prediction=model.predict(Xtr)
            row={**common,'primary_metric':'RMSE_ms',
                 'train_RMSE_ms':float(root_mean_squared_error(y_train,train_prediction)),
                 **regression_metrics(y_valid,prediction)}
        else:
            train_prediction=model.predict(Xtr)
            row={**common,'primary_metric':'F1',
                 'train_F1':float(f1_score(y_train,train_prediction,zero_division=0)),
                 **classification_metrics(y_valid,prediction)}
        rows.append(row)
        records.extend(make_records(task,name,Xva.index,y_valid,prediction,model,Xva))
        models[name]=model
        trace.append({'task':task,'model':name,'fit_ids':Xtr.index.tolist(),
                      'predict_ids':Xva.index.tolist(),'input_columns':Xtr.columns.tolist(),
                      'statistics_unchanged':unchanged,
                      'fitted_statistics':fitted_statistics,
                      'feature_names':model['pre'].get_feature_names_out().tolist()})
    return pd.DataFrame(rows),pd.DataFrame(records),models,trace

def make_records(task,name,ids,y,pred,model,X):
    """행 순서를 유지하는 제공 코드. 회귀 잔차는 실제-예측입니다."""
    probability=None
    if task=='classification':
        positive=list(model['model'].classes_).index(1)
        probability=model.predict_proba(X)[:,positive]
    records=[]
    for i,request_id in enumerate(ids):
        row={'request_id':request_id,'split':'valid','task':task,'model':name,
             'y_true':float(y.iloc[i]),'y_pred':float(pred[i]),
             'residual_ms':float(y.iloc[i]-pred[i]) if task=='regression' else None,
             'probability_failed':float(probability[i]) if probability is not None else None}
        records.append(row)
    return records

def select_models(regression,classification):
    """제공: 사전 지표로 정렬. 완전 동점이면 먼저 등록한 단순 후보를 우선합니다."""
    r=regression.sort_values('RMSE_ms',ascending=True,kind='stable').iloc[0]
    c=classification.sort_values('F1',ascending=False,kind='stable').iloc[0]
    return {'regression':{'model':r['model'],'metric':'RMSE_ms','value':float(r['RMSE_ms'])},
            'classification':{'model':c['model'],'metric':'F1','value':float(c['F1'])}}

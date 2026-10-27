"""CV·valid·해석. train 내부 CV와 외부 valid의 역할을 분리합니다."""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from course_support import cls_metrics, reg_metrics, snapshot, json_ready

def run_candidate(name, pipe, d, folds, task='classification'):
    """행 dict, fold DataFrame, 검증예측 DataFrame, train60으로 fit한 Pipeline 반환.
    CV는 clone으로 fold 안에서 fit; valid는 train60으로 별도 fit한 모델로만 예측.
    보관 test의 입력·정답은 받지 않습니다.
    """
    is_cls=(task=='classification')
    yt=d.yc_train if is_cls else d.yr_train
    yv=d.yc_valid if is_cls else d.yr_valid
    scorer='f1' if is_cls else 'neg_root_mean_squared_error'
    # TODO D7-07-E1: 위 계약과 README를 읽고 구현하세요.
    # raise NotImplementedError("TODO D7-07-E1")
    # fold마다 새 Pipeline을 학습하여 평가 점수와 훈련 점수를 함께 구합니다.
    cv=cross_validate(clone(pipe),d.X_train,yt,cv=folds,scoring=scorer,
        return_train_score=True,return_estimator=True,n_jobs=1,error_score='raise')
    scores=cv['test_score'];train_scores=cv['train_score']
    if not is_cls:
        scores=-scores;train_scores=-train_scores
    # valid 예측에는 train 60건 전체로 새로 학습한 모델만 사용합니다.
    fitted=clone(pipe).fit(d.X_train,yt)
    pred=fitted.predict(d.X_valid)
    metric='F1' if is_cls else 'RMSE_ms'
    valid=cls_metrics(yv,pred) if is_cls else reg_metrics(yv,pred)
    train=cls_metrics(yt,fitted.predict(d.X_train)) if is_cls else reg_metrics(yt,fitted.predict(d.X_train))
    row={'model':name,'task':task,'scaling':'none' if name in ['Tree','Forest','XGB'] else 'standard',
         'CV_mean':float(scores.mean()),'CV_std':float(scores.std(ddof=0)),
         'CV_metric':metric,'train_primary':train[metric],
         **{'valid_'+k:v for k,v in valid.items()},
         'params':json.dumps(json_ready(fitted['model'].get_params()),ensure_ascii=False,sort_keys=True)}
    fold_rows=[]
    for i,((a,b),score,tr,fold_model) in enumerate(zip(folds,scores,train_scores,cv['estimator']),1):
        expected=d.X_train.iloc[a][['input_chars','allowed_tools']].median().to_numpy()
        actual=fold_model['pre'].named_transformers_['num']['imputer'].statistics_
        # 이 검사는 모든 단계에 제공합니다. 06의 통계 감사에서도 같은 원리를 구현합니다.
        if not np.allclose(expected,actual): raise AssertionError('fold外の中央値を使用')
        fold_rows.append({'task':task,'model':name,'fold':i,'fit_n':len(a),'eval_n':len(b),
            'metric':metric,'score':float(score),'train_score':float(tr),'median_audit_passed':True})
    predictions=pd.DataFrame({'request_id':d.X_valid.index,'task':task,'model':name,
                              'actual':yv.to_numpy(),'predicted':pred})
    if is_cls:
        pindex=list(fitted.classes_).index(1)
        predictions['failure_probability']=fitted.predict_proba(d.X_valid)[:,pindex]
    else: predictions['residual_ms']=yv.to_numpy()-pred
    return row,pd.DataFrame(fold_rows),predictions,fitted

def importance_table(forest, d, include_permutation=False):
    """train MDI 8행. 응용이면 valid 원본 4열 순열 진단 4행 추가. 인과/특성선택 아님."""
    # TODO D7-07-E2: 위 계약과 README를 읽고 구현하세요.
    # raise NotImplementedError("TODO D7-07-E2")
    mdi=pd.DataFrame({'feature':forest['pre'].get_feature_names_out(),
        'method':'MDI','importance':forest['model'].feature_importances_,
        'repeat_std':np.nan,'evaluation_data':'train60','metric':'impurity_decrease'})
    if not include_permutation:
        return mdi
    # 원본 네 열을 하나씩 섞어 valid F1이 얼마나 떨어지는지 진단합니다.
    result=permutation_importance(forest,d.X_valid,d.yc_valid,scoring='f1',
        n_repeats=20,random_state=42,n_jobs=1)
    permutation=pd.DataFrame({'feature':d.X_valid.columns,'method':'permutation',
        'importance':result.importances_mean,'repeat_std':result.importances_std,
        'evaluation_data':'valid20 (diagnostic)','metric':'F1_drop'})
    return pd.concat([mdi,permutation],ignore_index=True)

def choose_by_cv(table, task='classification'):
    """F1 평균 최대 / RMSE 평균 최소. 정확히 동점이면 사전 후보 순서 우선.
    valid 점수는 함수에서 참조하지 않습니다. 선택점수는 최종 성능이 아닙니다.
    """
    # TODO D7-07-E3: 위 계약과 README를 읽고 구현하세요.
    # raise NotImplementedError("TODO D7-07-E3")
    if table.empty:
        raise ValueError('비교표가 비어 있습니다.')
    # idxmax/idxmin은 완전히 같은 점수일 때 먼저 선언된 행을 선택합니다.
    index=table['CV_mean'].idxmax() if task=='classification' else table['CV_mean'].idxmin()
    if task not in ('classification','regression'):
        raise ValueError(task)
    return table.loc[index,'model']

"""입력/분할/출력의 계약 검사. 숫자 점수뿐 아니라 원인과 전제조건을 확인합니다."""
import numpy as np
from course_io import check_features,validate_k,snapshot

def before_model(estimator,X_train,y_train,X_valid,y_valid):
    Xtr=check_features(X_train,training=True); Xva=check_features(X_valid)
    if not Xtr.index.equals(y_train.index) or not Xva.index.equals(y_valid.index):
        raise ValueError('X와 y의 ID/순서가 다릅니다.')
    if set(Xtr.index)&set(Xva.index):
        raise ValueError('train/valid ID 교집합이 있습니다.')
    if hasattr(estimator,'n_neighbors'): validate_k(estimator.n_neighbors,len(Xtr))
    return Xtr,Xva

def check_finite(pred,n):
    a=np.asarray(pred)
    if a.shape!=(n,) or not np.isfinite(a).all():
        raise ValueError('예측 shape 또는 NaN/inf 오류')

def prediction_contract(model,X):
    """새 입력의 필수 열 검사→transform→predict. 8열/유한값/통계 불변을 확인한다.
    반환: n_rows,n_features,statistics_unchanged 딕셔너리. 통계가 바뀌면 AssertionError.
    missing/extra 열은 임의로 채우거나 조용히 삭제하지 않고 ValueError로 알린다.
    """
    # check_features는 열 누락, 금지 열, 잘못된 숫자와 중복 ID를 먼저 막습니다.
    checked=check_features(X)
    before=snapshot(model['pre'])
    transformed=model['pre'].transform(checked)
    if transformed.shape!=(len(checked),8) or not np.isfinite(transformed).all():
        raise ValueError('전처리 결과는 유한한 값으로 된 8열이어야 합니다.')
    prediction=model.predict(checked)
    check_finite(prediction,len(checked))
    unchanged=before==snapshot(model['pre'])
    if not unchanged:
        raise AssertionError('새 입력 예측 중 train 통계가 바뀌었습니다.')
    return {'n_rows':len(checked),'n_features':transformed.shape[1],
            'statistics_unchanged':unchanged}

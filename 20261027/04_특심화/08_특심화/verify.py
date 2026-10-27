"""실습08: 앞서 작성한 실습07 프로젝트의 출력 검증. 새 프로젝트를 만들지 않습니다."""
from pathlib import Path
import sys
import argparse
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
# PROJECT=ROOT/'ex07_project'/'03_심화'/'day7_ensemble'
# 현재 자료의 실제 폴더 이름을 기준으로 실습07 프로젝트를 찾습니다.
PROJECT=Path(__file__).resolve().parents[1]/'07_특심화'/'day7_ensemble'
sys.path.insert(0,str(PROJECT))
from course_support import load_data, write_json

def verify_cv_table(scores):
    """모델·task마다 fold1~5,48/12,유한점수, 분류F1[0,1]/회귀RMSE>=0 검사."""
    # TODO D7-08-V1: 위 계약과 README를 읽고 구현하세요.
    # raise NotImplementedError("TODO D7-08-V1")
    required={'task','model','fold','fit_n','eval_n','metric','score'}
    if not required.issubset(scores.columns):
        raise ValueError('CV 표의 필수 열이 없습니다.')
    for (task,_),group in scores.groupby(['task','model'],sort=False):
        if group['fold'].tolist()!=[1,2,3,4,5]:
            raise ValueError('각 후보에는 1~5 fold가 한 번씩 있어야 합니다.')
        if not ((group['fit_n']==48).all() and (group['eval_n']==12).all()):
            raise ValueError('각 fold는 학습48건, 평가12건이어야 합니다.')
        if not np.isfinite(group['score']).all():
            raise ValueError('점수는 유한한 숫자여야 합니다.')
        if task=='classification':
            if not ((group['metric']=='F1').all() and group['score'].between(0,1).all()):
                raise ValueError('분류 점수는 0~1 사이 F1이어야 합니다.')
        elif task=='regression':
            if not ((group['metric']=='RMSE_ms').all() and (group['score']>=0).all()):
                raise ValueError('회귀 점수는 0 이상의 RMSE여야 합니다.')
        else:
            raise ValueError(f'알 수 없는 task: {task}')
    return True

def verify_predictions(predictions,d):
    """각 모델/task당 valid20개 순서, 실제 정답 일치, test ID 미포함 검사."""
    # TODO D7-08-V2: 위 계약과 README를 읽고 구현하세요.
    # raise NotImplementedError("TODO D7-08-V2")
    required={'request_id','task','model','actual','predicted'}
    if not required.issubset(predictions.columns):
        raise ValueError('예측표의 필수 열이 없습니다.')
    expected_ids=list(d.X_valid.index)
    test_ids=set(d.test_ids)
    for (task,_),group in predictions.groupby(['task','model'],sort=False):
        if group['request_id'].tolist()!=expected_ids:
            raise ValueError('valid 20건의 ID와 순서가 다릅니다.')
        if set(group['request_id']) & test_ids:
            raise ValueError('보관 test ID가 예측표에 들어갔습니다.')
        expected=d.yc_valid if task=='classification' else d.yr_valid if task=='regression' else None
        if expected is None or not np.allclose(group['actual'].to_numpy(),expected.to_numpy()):
            raise ValueError('실제 정답이 원본 valid와 다릅니다.')
        if not np.isfinite(group['predicted']).all():
            raise ValueError('예측값은 유한한 숫자여야 합니다.')
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=PROJECT)
    args=p.parse_args(); out=args.project/'outputs'
    d=load_data(args.project/'data')
    summary=json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
    if summary['test_evaluated'] is not False: raise ValueError('test 평가 금지')
    scores=pd.read_csv(out/'cv_scores.csv',encoding='utf-8-sig')
    predictions=pd.read_csv(out/'validation_predictions.csv',encoding='utf-8-sig')
    result={'cv_table':verify_cv_table(scores),'predictions':verify_predictions(predictions,d),
            'test_not_evaluated':True}
    write_json(out/'submission_check.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

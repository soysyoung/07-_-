"""특성 선택·전처리·모델 생성. 아직 학습하지 않은 Pipeline을 만듭니다."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from course_support import NUM, CAT, FEATURES
from model_config import new_classifier, new_regressor

def make_preprocessor(scale='none'):
    """num: median (+standard 선택), cat: 최빈값 -> dense OneHot(ignore)."""
    # TODO D7-07-P1: 위 계약과 README를 읽고 구현하세요.
    # raise NotImplementedError("TODO D7-07-P1")
    if scale not in ('none', 'standard'):
        raise ValueError("scale은 none 또는 standard여야 합니다.")
    # 숫자는 빈칸을 중앙값으로 채우고, 필요한 모델에만 표준화를 적용합니다.
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale == 'standard':
        numeric_steps.append(('scaler', StandardScaler()))
    category = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', Pipeline(numeric_steps), NUM),
        ('cat', category, CAT),
    ], remainder='drop')

def candidates(include_previous=False, task='classification'):
    """기본 분류 Dummy/Tree/Forest/XGB. 응용에 Logistic/KNN3, 회귀는 6후보.
    Dummy와 전날 모델만 scale=standard. 각 후보는 서로 다른 새 객체여야 합니다.
    """
    # TODO D7-07-P2: 위 계약과 README를 읽고 구현하세요.
    # raise NotImplementedError("TODO D7-07-P2")
    if task == 'classification':
        names = ['Dummy']
        if include_previous:
            names += ['Logistic', 'KNN3']
        names += ['Tree', 'Forest', 'XGB']
        factory = new_classifier
    elif task == 'regression':
        names = ['Dummy', 'Linear', 'KNN15', 'Tree', 'Forest', 'XGB']
        factory = new_regressor
    else:
        raise ValueError(task)
    # 거리나 선형 계산을 쓰는 모델은 숫자 크기를 맞추고, 트리 모델은 원래 크기를 씁니다.
    unscaled = {'Tree', 'Forest', 'XGB'}
    return {name: Pipeline([
        ('pre', make_preprocessor('none' if name in unscaled else 'standard')),
        ('model', factory(name)),
    ]) for name in names}

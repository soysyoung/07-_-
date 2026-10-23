"""05 기본필수: qa와 summary를 서로 다른 0/1 열로 표현합니다."""
from sklearn.preprocessing import OneHotEncoder

train = [["qa"], ["summary"], ["qa"]]
valid = [["summary"], ["qa"]]

# 범주형 문자열을 각 범주에 대응하는 0과 1의 숫자 열로 변환합니다.
# sparse_output=False는 결과를 바로 볼 수 있는 숫자 표로 받는 설정입니다.
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(train)

# TODO B05-1: encoder.transform(valid).tolist()로 실제 변환 결과를 구하세요.
# valid_ready = [[0.0, 0.0], [0.0, 0.0]]
# train에서 학습한 범주 순서를 기준으로 valid를 원-핫 인코딩합니다.
# tolist()는 NumPy 배열을 일반 파이썬 리스트로 변환합니다.
valid_ready = encoder.transform(valid).tolist()

print("열 순서:", encoder.categories_[0].tolist())
print("valid 변환:", valid_ready)

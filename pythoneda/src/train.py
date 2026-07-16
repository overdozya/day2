"""
프로그램명: [Day 2 평가] scikit-learn Pipeline 기반 기계학습 모델 구축 실습
작성자: 2조 [안동선, 김민솔, 황민진, 권태현, 정승우]
작성일: 2026-07-16
설명:
    - 데이터 스케일링과 학습 모델을 통합하여 일관된 분류 모델 파이프라인을 학습시킵니다.
    - 주요 프로세스:
      1) StandardScaler와 LogisticRegression 분류기를 단일 sklearn Pipeline 구조로 묶음
      2) 데이터 누수(Data Leakage) 원천 차단을 전제로 feature 및 target 데이터셋 분할 가공
      3) 학습 및 평가 데이터 분할(8:2), 모델 학습 수행 및 Test 데이터 기준 분류 보고서(F1-score) 출력
      4) 학습 완료된 파이프라인 객체와 레이블 정보를 joblib 바이너리 파일로 영구 저장
"""

# ==========================================
# 1. 머신러닝 프로세스를 위한 scikit-learn 및 데이터 라이브러리 임포트
# ==========================================
import pandas as pd             # 정제 데이터 수급용 모듈
import numpy as np              # 고성능 행렬 연산 및 결측값 식별을 위한 모듈
from sklearn.model_selection import train_test_split  # 전체 데이터를 학습셋(Train)과 테스트셋(Test)으로 분리하는 모듈
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 데이터 정규화 및 클래스 레이블 수치 변환 모듈
from sklearn.linear_model import LogisticRegression  # 이진 분류를 풀기 위한 대표적인 선형 분류 알고리즘 모듈
from sklearn.pipeline import Pipeline  # 데이터 가공(Scaler)과 학습 모델(Clf)을 하나로 결합하는 워크플로우 제어 모듈
from sklearn.metrics import accuracy_score, f1_score, classification_report  # 모델 다각도 정량 평가를 위한 성능 지표 계산 모듈
import joblib                   # 학습이 끝난 파이프라인 모델 객체를 고속으로 디스크에 저장 및 로드하는 라이브러리
import os                       # 생성된 모델 산출물을 기록할 로컬 폴더(models/) 제어 모듈

# ==========================================
# 2. 데이터셋 전처리 및 피처-타겟 분리 함수
# ==========================================
def prepare_data(filepath: str = "data/cleaned.parquet"):
    """
    정제된 와인 Parquet 데이터를 로드하여 타겟(Y) 클래스를 인코딩하고,
    피처(X)를 선별한 뒤, 8:2 비율로 데이터를 학습 및 테스트용으로 최종 분할합니다.
    """
    df = pd.read_parquet(filepath)
    
    # B) Wine Quality 이진 분류의 타겟 컬럼 지정
    target_col = "quality_binary"
    
    # 범주형 타겟 변수가 혹시라도 문자열 형태일 경우를 대비해 정수형(0, 1 등)으로 변환해 주는 변환기를 가동합니다.
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    
    # [매우 중요] 데이터 누수(Data Leakage) 원천 차단:
    # 타겟값인 'quality_binary' 외에, 원본 연속형 점수인 'quality' 컬럼도 피처 목록에서 완전히 제외합니다.
    # 만약 원본 'quality'를 학습 피처에 그대로 두면, 모델은 수학적 패턴을 배우는 대신 
    # 'quality가 7 이상인지'만을 기계적으로 판단하는 무의미한 지름길을 학습하게 됩니다.
    exclude_cols = {target_col, "quality"}
    
    # 피처로 사용할 수치형 데이터 컬럼만 골라 리스트로 묶습니다.
    feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in exclude_cols]
    
    # scikit-learn 모델에 주입하기 위해 Pandas DataFrame 구조를 순수 Numpy 다차원 배열(NDArray)로 전환합니다.
    X = df[feature_cols].values
    
    # 학습 데이터와 검증용 테스트 데이터를 8:2 비율로 분리합니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,     # 동일한 분할 결과를 재현하기 위한 난수 고정값 설정
        stratify=y           # 클래스 불균형 문제를 해소하기 위해 원본 비율(예: 좋은 와인과 나쁜 와인 비율)을 분할 후에도 동일하게 유지
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"클래스: {le.classes_}")
    return X_train, X_test, y_train, y_test, le, feature_cols

# ==========================================
# 3. 모델 전처리 + 모델 결합 파이프라인 빌드 함수
# ==========================================
def build_pipeline() -> Pipeline:
    """
    학습 데이터셋 기준으로 학습된 스케일러 파라미터를 그대로 테스트 단계에 연계 적용하기 위해,
    StandardScaler와 LogisticRegression을 하나의 '파이프라인' 구조로 일체화합니다.
    """
    return Pipeline([
        # 1단계: 변수들의 단위를 표준화하는 스탠다드 스케일러 장착 (평균 0, 표준편차 1로 스케일링)
        ("scaler", StandardScaler()),
        # 2단계: 최적화를 위한 선형 분류 모델 탑재 (경사하강법 최대 반복횟수를 1000회로 확장)
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

# ==========================================
# 4. 파이프라인 학습 실행 함수
# ==========================================
def train_pipeline(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """
    구축된 파이프라인 모델에 학습용 피처(X_train)와 정답지(y_train)를 주입하여 스케일링 기준 학습 및 분류기 피팅을 시작합니다.
    """
    pipeline.fit(X_train, y_train)
    print("파이프라인 학습 완료!")
    return pipeline

# ==========================================
# 5. 성능 평가 및 분류 보고서 출력 함수
# ==========================================
def evaluate_pipeline(pipeline: Pipeline, X_test, y_test, label_encoder):
    """
    학습에 사용되지 않은 평가용 데이터(X_test)를 활용하여 정밀도를 검증하고,
    오차 정밀도, 재현율, 그리고 불균형 데이터에 적합한 가중 F1-score 지표를 산출해 상세 리포트를 화면에 출력합니다.
    """
    # 학습이 완료된 파이프라인의 분류 모델로 예측값 생성
    y_pred = pipeline.predict(X_test)
    
    # 단순 예측 정확도 산출
    acc = accuracy_score(y_test, y_pred)
    
    # 데이터 분포 비대칭성을 감안한 가중합 가중치 F1-score 계산
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(f"\n=== Test Accuracy: {acc:.4f} ===")
    print(f"F1-score: {f1:.4f}")
    print("\n=== Classification Report ===")
    # 정밀도, 재현율 등 세부 정량 통계를 텍스트로 시각화해 주는 내장 리포트 가동
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))
    return acc, f1

# ==========================================
# 6. 평가 정확도 데이터 저장 함수
# ==========================================
def save_accuracy(acc: float, f1: float, filepath: str = "models/accuracy.txt") -> None:
    """
    자동 리포트 작성 스크립트(generate_report.py)에서 읽어갈 수 있도록,
    정확도와 F1-score의 요약 텍스트 파일(accuracy.txt)을 생성하여 디스크에 저장합니다.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(f"- Accuracy: {acc:.4f}\n- F1-score (weighted): {f1:.4f}\n")
    print(f"정확도 저장: {filepath}")

# ==========================================
# 7. 파이프라인 직렬화(저장) 함수
# ==========================================
def save_pipeline(pipeline, label_encoder, feature_cols, filepath: str = "models/pipeline.joblib") -> None:
    """
    차후 실서비스나 테스트에서 즉시 재활용할 수 있도록,
    학습된 파이프라인 모델, 인코더 정보 및 피처명을 하나로 패키징하여 joblib 파일로 바이너리 압축 저장합니다.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols
    }, filepath)
    print(f"파이프라인 저장: {filepath}")

# ==========================================
# 8. 메인 실행 제어 엔진
# ==========================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le, feature_cols = prepare_data() # 데이터셋 가공
    pipeline = build_pipeline()                                        # 파이프라인 생성
    pipeline = train_pipeline(pipeline, X_train, y_train)              # 모델 피팅(학습)
    acc, f1 = evaluate_pipeline(pipeline, X_test, y_test, le)          # 성능 평가 수행
    save_accuracy(acc, f1)                                             # 평가지표 텍스트화 보존
    save_pipeline(pipeline, le, feature_cols)                         # 파이프라인 영구 저장
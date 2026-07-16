"""
프로그램명: [Day 2 평가] 전체 데이터 분석 파이프라인 자동화 및 마크다운 리포트 생성 실습
작성자: 2조 [안동선, 김민솔, 황민진, 권태현, 정승우]
작성일: 2026-07-16
설명:
    - 전처리, 시각화, 모델 학습 모듈을 순차 가동시키고 연계 리포트를 생성하는 제어 스크립트입니다.
    - 주요 프로세스:
      1) subprocess를 활용하여 앞선 3개 동작 모듈(eda, visualize, train)을 안전하게 순차 트리거
      2) 실행 결과로 확보된 기술통계, 상관계수, t-test p-value 및 머신러닝 성능 텍스트 종합
      3) f-string 마크다운 구조 렌더링 후 최종 report.md 아티팩트 자동 작성 배포
"""


# ==========================================
# 1. 시스템 제어 및 데이터 처리를 위한 라이브러리 임포트
# ==========================================
import subprocess  # 외부 터미널 명령어를 파이썬 내부 코드 흐름 상에서 백그라운드로 실행하기 위한 모듈
import sys         # 시스템 전용 매개변수 및 파이썬 엔진 실행 경로(sys.executable) 정보 획득용 모듈
from pathlib import Path  # 파일/폴더 경로를 단순 문자열이 아닌 독립 객체로 안전하게 다루기 위한 라이브러리
import pandas as pd  # 구조화된 2차원 표 형식 데이터(Parquet)를 가공 및 요약하기 위한 데이터분석 라이브러리
import numpy as np   # 행렬 연산 및 수치형 데이터 타입(np.number) 필터링에 사용하는 수학 라이브러리
from scipy.stats import ttest_ind  # 두 집단의 평균값 차이가 유의미한지 통계적으로 검증하기 위한 과학용 라이브러리

# ==========================================
# 2. 하위 모듈 실행기 (Subprocess 제어 함수)
# ==========================================
def run_step(cmd: list[str], description: str) -> bool:
    """
    주어진 터미널 명령어를 백그라운드로 안전하게 독립 실행하고, 
    그에 따른 성공/실패 여부를 불리언(True/False) 타입으로 반환합니다.
    """
    print(f"\n=== {description} ===")
    
    # subprocess.run을 사용하여 터미널 명령을 수행하고, 표준 출력(stdout) 및 표준 에러(stderr)를 문자열로 수집합니다.
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 실행 시 생성된 출력이 존재할 경우, 화면 혼잡을 막기 위해 상위 500자만 슬라이싱하여 요약 출력합니다.
    if result.stdout:
        print(result.stdout[:500])
        
    # 명령어의 종료 코드(Return Code)가 0이 아니라면 내부 치명적 오류가 발생한 것으로 판단합니다.
    if result.returncode != 0:
        print(f"[실패] {description}")
        # 오류 상세 파악을 위해 에러 로그(stderr)를 최대 300자 출력합니다.
        print(result.stderr[:300])
        return False
        
    # 성공적으로 완료(종료 코드 0) 시 True 반환
    print(f"[성공] {description}")
    return True

# ==========================================
# 3. 마크다운 종합 리포트 생성기
# ==========================================
def generate_report():
    """
    앞선 단계들에서 저장한 통계 수치, 검정 결과값 및 머신러닝 성능 지표를
    모두 취합하여 하나의 마크다운 파일(report.md)로 자동 출력 및 바인딩합니다.
    """
    # 전처리 단계(eda.py)의 결과물인 Parquet 형식의 정제 데이터를 불러옵니다.
    df = pd.read_parquet("data/cleaned.parquet")
    
    # 수치형 컬럼들만 리스트로 필터링하여 통계 요약 및 상관관계 분석을 준비합니다.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 수치 변수들에 대한 주요 기술통계량(평균, 편차, 사분위수 등)을 연산합니다.
    desc = df[numeric_cols].describe()
    
    # 수치 변수들 간의 피어슨 상관계수 행렬을 계산합니다.
    corr = df[numeric_cols].corr()
    
    # B) Wine Quality 데이터셋 구조 지정을 위한 타겟 및 독립변수 할당
    target_col = "quality_binary" # 예측 타겟: 고품질 여부(0 또는 1)
    value_col = "alcohol"         # 검정 대상: 와인의 알코올 도수
    
    # 템플릿 스트링을 사용하여 마크다운 문서의 뼈대를 생성합니다.
    report = f"""# 데이터 분석 리포트
## 데이터셋 정보
- 파일: data/cleaned.parquet
- 행 수: {len(df)}
- 컬럼 수: {len(df.columns)}
- Target 컬럼: {target_col}

## 기초 통계
{desc.to_string()}

## 상관계수 행렬
{corr.to_string()}

## t-test 결과 (두 그룹 비교)
"""
    # t-test 수행 및 검정 p-value에 대한 논리적 의사결정 구간
    try:
        # 타겟 클래스 고유값 추출 (0: 저품질, 1: 고품질)
        group_labels = df[target_col].unique()
        
        # 유효한 집단이 최소 2개 이상일 때 정상적인 검정 수행 시작
        if len(group_labels) >= 2:
            # 타겟값이 첫 번째 그룹에 속하는 데이터의 알코올 도수 데이터셋 확보 (결측치는 즉각 필터링)
            g1 = df[df[target_col] == group_labels[0]][value_col].dropna()
            # 타겟값이 두 번째 그룹에 속하는 데이터의 알코올 도수 데이터셋 확보
            g2 = df[df[target_col] == group_labels[1]][value_col].dropna()
            
            # 독립표본 t-검정 수행 (equal_var=False 설정을 통한 이분산 웰치의 t-test 적용)
            stat, p_value = ttest_ind(g1, g2, equal_var=False)
            
            # 검정 결과 및 통계적 유의성 해석 문구를 마크다운 문자열에 누적시킵니다.
            report += f"""- 비교 대상: {target_col} ({group_labels[0]} vs {group_labels[1]})
- 측정 컬럼: {value_col}
- t-statistic: {stat:.4f}
- p-value: {p_value:.4f}
- 결론: {"유의미한 차이 있음" if p_value < 0.05 else "유의미한 차이 없음"} (p={'<' if p_value < 0.05 else '>='} 0.05)
"""
    except Exception as e:
        # 연산 과정에 결함 발생 시 중단되지 않도록 에러 트레이스를 마크다운 문서 내에 기록합니다.
        report += f"- t-test 오류: {e}\n"
        
    # ML 모델 학습 결과 반영 단계
    report += "\n## ML 모델 성능\n"
    pipeline_file = Path("models/pipeline.joblib")
    
    # 학습 파이프라인 아티팩트의 물리적 존재 여부를 확인합니다.
    if pipeline_file.exists():
        acc_path = Path("models/accuracy.txt")
        # 저장된 텍스트 형식의 정확도 지표 파일을 읽어 리포트 꼬리에 붙입니다.
        if acc_path.exists():
            acc = acc_path.read_text().strip()
            report += f"{acc}\n"
            report += f"- 모델 파일: models/pipeline.joblib\n"
    else:
        report += "- 모델 파일 없음\n"
        
    # 완성된 리포트 스트링을 디스크에 한글 깨짐이 없도록 UTF-8 포맷으로 안전하게 저장합니다.
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("report.md 생성 완료!")

# ==========================================
# 4. 전체 제어 메인 엔진 블록
# ==========================================

if __name__ == "__main__":
    steps = [
        (["python", "-m", "src.eda"], "1/3 EDA"),
        (["python", "-m", "src.visualize"], "2/3 시각화"),
        (["python", "-m", "src.train"], "3/3 ML 학습"),
    ]
    # 선언된 세 개의 단계를 순차적으로 순회 구동합니다.
    for cmd, desc in steps:
        # 만약 한 단계라도 실패(False 반환)할 경우 전체 시스템 신뢰성을 위해 프로그램을 강제 즉시 종료합니다.
        if not run_step(cmd, desc):
            sys.exit(1)
            
    # 모든 백그라운드 연계가 정상 완료되면, 최종적으로 종합 통계 리포트 작성을 트리거합니다.
    generate_report()
    print("\n전체 파이프라인 완료! report.md 를 확인하세요.")
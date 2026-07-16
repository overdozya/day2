"""
프로그램명: [Day 2 평가] Seaborn 및 Plotly 기반 정적·인터랙티브 데이터 시각화 실습
작성자: 2조 [안동선, 김민솔, 황민진, 권태현, 정승우]
작성일: 2026-07-16
설명:
    - 정제된 와인 데이터를 바탕으로 분포 시각화 및 변수 간의 관계 시각화를 실행합니다.
    - 주요 프로세스:
      1) Matplotlib/Seaborn 기반의 수치형 변수 분포(히스토그램, KDE, Boxplot) 정적 PNG 시각화 및 저장
      2) Plotly Express를 활용한 인터랙티브 산점도 행렬(Scatter Matrix) 생성 및 HTML 바인딩
      3) 피어슨 상관계수 행렬을 기반으로 한 Plotly 대화형 상관관계 히트맵(Heatmap) 생성
"""

# ==========================================
# 1. 시각화 및 데이터 핸들링을 위한 라이브러리 임포트
# ==========================================
import pandas as pd             # Parquet 형식의 가공 데이터를 읽어와 구조화하기 위한 모듈
import seaborn as sns           # 통계적 데이터 분석 및 분포 시각화에 특화된 정적 시각화 라이브러리
import matplotlib.pyplot as plt # 차트 객체 생성, 크기 조정, 이미지 파일 저장 등을 제어하는 기본 시각화 라이브러리
import os                       # 차트 이미지를 내보낼 저장 폴더(visualizations/)가 없을 시 생성하기 위한 시스템 제어 모듈
import plotly.express as px     # 코드 한 줄로 화려한 마우스 반응형(대화형) 인터랙티브 차트를 구현하는 라이브러리
import plotly.io as pio         # Plotly의 전역 환경설정 및 엔진 입출력 제어를 위한 모듈

# ==========================================
# 2. 전역 설정 영역
# ==========================================
# Plotly HTML 저장 시, 레이텍(LaTeX) 수식 렌더러인 MathJax 로딩을 꺼둠으로써 파일 용량을 대폭 경량화합니다.
pio.defaults.mathjax = None 

# ==========================================
# 3. 데이터 로딩 함수
# ==========================================
def load_clean_data(filepath: str = "data/cleaned.parquet") -> pd.DataFrame:
    """
    이전 단계(eda.py)에서 결측치를 모두 정제해 둔 고효율 Parquet 압축 데이터를
    Pandas 데이터프레임 구조로 고속 로딩합니다.
    """
    return pd.read_parquet(filepath)

# ==========================================
# 4. Seaborn 정적 차트 생성 함수 (히스토그램 & 박스플롯)
# ==========================================
def plot_static_charts(df: pd.DataFrame, output_dir: str = "visualizations") -> None:
    """
    수치형 데이터 컬럼들을 식별하여 정적 이미지 형식(PNG)의 분포 차트를 저장합니다.
    """
    # 저장할 디렉토리가 물리적으로 없으면 신규 생성하고, 있으면 통과(exist_ok=True)합니다.
    os.makedirs(output_dir, exist_ok=True)
    
    # 실수형(float64) 및 정수형(int64)인 정량적 수치 변수들의 컬럼 이름만 모아 리스트화합니다.
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    # 너무 많은 차트가 동시에 그려지는 것을 예방하기 위해, 분석 편의상 상위 3개 변수만 슬라이싱하여 제한합니다.
    if len(numeric_cols) > 3:
        numeric_cols = numeric_cols[:3]
        
    # 제한된 상위 수치 변수들을 순회하며 차트 생성을 시작합니다.
    for col in numeric_cols:
        # 가로 8인치, 세로 4인치 규격의 캔버스 도화지를 선언합니다.
        plt.figure(figsize=(8, 4))
        
        # 1) 히스토그램 생성: 막대 분포 위에 부드러운 확률 밀도 함수 커브(kde=True)를 함께 덧그립니다.
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        
        # 차트 제목이나 축 라벨이 잘리는 현상을 막기 위해 여백을 타이트하게 조절하여 PNG로 내보냅니다.
        plt.savefig(f"{output_dir}/hist_{col}.png", bbox_inches="tight")
        plt.close() # 메모리 점유 방지를 위해 열려 있는 pyplot 피겨 객체를 닫아줍니다.
        
        # 2) 박스플롯 생성: 데이터의 고유 값 개수가 2개 초과인 경우에만 의미가 있으므로 조건을 부여합니다.
        if df[col].nunique() > 2:
            plt.figure(figsize=(8, 4))
            
            # 수평 구조의 아웃라이어 식별용 가로형 박스플롯 차트를 생성합니다.
            sns.boxplot(x=df[col])
            plt.title(f"Box Plot of {col}")
            
            # 조절된 여백을 유지하며 박스플롯 PNG 저장
            plt.savefig(f"{output_dir}/box_{col}.png", bbox_inches="tight")
            plt.close() # 사용된 자원 반환
            
    print(f"Seaborn 정적 차트 저장 완료: {output_dir}/")

# ==========================================
# 5. Plotly 대화형 차트 생성 함수 (산점도 행렬 & 상관 히트맵)
# ==========================================
def plot_interactive_charts(df: pd.DataFrame, output_dir: str = "visualizations") -> None:
    """
    다변량 데이터의 상관 상태를 확인하기 위해 브라우저 상에서 확대/축소 및 마우스 호버(Hover)가
    가능한 대화형 인터랙티브 차트(HTML 및 PNG)를 렌더링하고 보존합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    # 1) 산점도 행렬(Scatter Matrix) 구성: 상위 4개 수치형 변수들 간의 조합을 평면 좌표에 대입합니다.
    fig_scatter = px.scatter_matrix(
        df, dimensions=numeric_cols[:4],
        title="Scatter Matrix",
        template="plotly_white" # 흰색 바탕의 정돈된 현대식 스타일 레이아웃 적용
    )
    # 웹 브라우저에서 상호작용 가능한 무설치 독립형 대형 HTML 파일로 내보냅니다.
    fig_scatter.write_html(f"{output_dir}/scatter_matrix.html")
    
    # 문서용 PNG 형식의 정적 아티팩트로도 내보내기를 시도합니다. (kaleido 라이브러리 필요)
    try:
        fig_scatter.write_image(f"{output_dir}/scatter_matrix.png")
    except Exception as e:
        print(f"[경고] scatter_matrix.png 저장 실패: {e}")
        
    # 2) 상관계수 산출 및 대화형 히트맵 구성
    corr = df[numeric_cols].corr() # 전체 수치형 컬럼 간의 선형 상관 계수 산정
    
    # imshow 함수로 다차원 상관계수 행렬을 바둑판 형태의 타일 히트맵으로 전환합니다.
    fig_heat = px.imshow(
        corr, 
        text_auto=".2f",                 # 각 타일 내부 칸마다 소수점 둘째 자리까지 상관 계수 수치 자동 매핑
        aspect="auto",                   # 정사각형이 아닌 화면 비율에 맞추어 유동적으로 정렬
        title="Correlation Heatmap",
        template="plotly_white",
        color_continuous_scale="RdBu_r"  # 상관 관계를 직관적으로 파악할 수 있도록 Red(양의 상관)와 Blue(음의 상관) 대조 색상표 적용
    )
    # 인터랙티브 전용 HTML 아티팩트 보존
    fig_heat.write_html(f"{output_dir}/heatmap.html")
    
    # 보고서 삽입용 정적 이미지(PNG) 내보내기 시도
    try:
        fig_heat.write_image(f"{output_dir}/heatmap.png")
    except Exception as e:
        print(f"[경고] heatmap.png 저장 실패: {e}")
        
    print(f"Plotly 인터랙티브 차트 저장 완료: {output_dir}/")

# ==========================================
# 6. 모듈 단독 실행 블록
# ==========================================
if __name__ == "__main__":
    df = load_clean_data()            # 정제 데이터 수급
    plot_static_charts(df)            # Seaborn 기반 분포 분석 시각화 구동
    plot_interactive_charts(df)       # Plotly 기반 변수 상관 시각화 구동
    print("모든 시각화 완료!")
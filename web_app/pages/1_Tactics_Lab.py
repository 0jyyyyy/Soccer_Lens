import streamlit as st

st.set_page_config(page_title="Soccer Lens - Tactics_Lab", page_icon="🔍", layout="wide")

# main.py를 거치지않고 이 페이지를 새로고침했을때 에러 방지용 기본값
if "selected_season" not in st.session_state:
  st.session_state.selected_season = "2025/26"
if "selected_league" not in st.session_state:
  st.session_state.selected_league = "Ligue 1"

# 상세 리포트 클릭 상태를 저장할 전용 데이터 (최초에는 None)
if "viewing_tactics_team" not in st.session_state:
  st.session_state.viewing_tactics_team = None

# 메인에서 공유해 준 데이터 값 가져오기
current_season = st.session_state.selected_season
current_league = st.session_state.selected_league

st.title("🔍 Tactics Lab (경기 분석)")
st.caption(f"📅 {current_season}  |  🌍 {current_league}")
st.markdown("---")

#======================================================
# 핵심 로직 : 유저가 메인에서 선택한 값에 따른 조건부 렌더링
#======================================================

# 리그앙(Ligue 1)
if current_season == "2025/26" and current_league == "Ligue 1":
  st.subheader("🇫🇷 프랑스 리그앙(Ligue 1) 전술 분석 리포트 목록")

  col1, col2, col3 = st.columns(3)

  with col1:
    with st.container(border=True):
      st.markdown("### 🔴 🟡 RC 랑스 (RC Lens)")
      st.markdown("**돌풍의 주역: 컴팩트 전방 압박 메커니즘**")
      st.caption("📅 업데이트: 2026년 6월 | 📊 표본: 3 Match")
      if st.button("📄 랑스 딥다이브 리포트 읽기", key="btn_lens"):
        st.session_state.viewing_tactics_team = "rc_lens"
        st.rerun()

  st.markdown("---")

  #===============================================
  # 본문 출력 구역
  #===============================================
  if st.session_state.viewing_tactics_team == "rc_lens":
    # 본문 닫기 버튼
    if st.button("❌ 본문 리포트 닫기", key="close_lens"):
      st.session_state.viewing_tactics_team = None
      st.rerun()
            
    st.header("🏟️ RC 랑스 (RC Lens) Deep Dive 전술 리포트")
    st.caption("분석 총괄: 0jyyyyy | 분석 프레임워크: Soccer Lens 1.0")
        
    # 경기분석 탭 분할 (표본 경기 3~5개만 집중 스캐닝하는 전략)
    match_tab1, match_tab2 = st.tabs([
    "⚔️ vs 파리 생제르맹 (전방 압박 분석)", 
    "⚔️ vs 마르세유 (빌드업 대형 분석)"
    ])
        
    # [표본 1] PSG전 리포트 본문
    with match_tab1:
      st.markdown("### 📝 RC 랑스 vs 파리 생제르맹 전술 분석")
            
      # 1단계: 주관적 통찰 던지기
      st.subheader("1️⃣ 분석가 Insight Log (주관적 통찰)")
      st.warning("""
      **"이날 랑스의 승리 요인은 감정적인 투지가 아니라, PSG의 후방 빌드업 체계를 완벽하게 마비시킨 '컴팩트한 전방 압박 타이밍'에 있었습니다."** 랑스는 3-4-3 대형을 유지하며 PSG의 볼란치 라인으로 들어오는 패스 길목을 타이트하게 통제했습니다.
      """)
            
      # 2단계: 객관적 데이터 지표 뒷받침 (우회 치트키 구역)
      st.subheader("2️⃣ 객관적 지표 검증 (Data Evidence)")
      col_data1, col_data2 = st.columns(2)
      with col_data1:
          st.info("📊 경기 시간대별 xG(기대득점) 추이 (출처: FotMob 등 외부 캡처 이미지 매핑 예정 구역)")
      with col2:
          st.info("🗺️ 팀 평균 압박 위치 및 패스 네트워크 맵 (출처: 웅스탯 등 외부 캡처 이미지 매핑 예정 구역)")
          
      # 3단계: 내 무기인 비전 분석 기술로 최종 증명 (YOLO / ByteTrack 클립)
      st.subheader("3️⃣ 비전 분석 최종 증명 (Soccer Lens Showcase)")
      st.success("🔥 아래 영상은 분석가가 직접 개발한 OpenCV + YOLOv8 + ByteTrack 모듈로 선수들의 움직임을 추적한 크롭 파일입니다.")
      st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ") # 실제 분석 영상 유튜브 URL 매핑
      st.caption("🔍 비전 분석 포인트: 영상 7초 지점, PSG 수비진 소유 시 랑스 전방 포워드 3명의 간격이 보라색 태깅 박스로 12m 이내 유지됨.")

      # [표본 2] 마르세유전 리포트 본문
      with match_tab2:
          st.markdown("### 📝 RC 랑스 vs 마르세유 전술 분석")
          st.info("🚧 마르세유전 OpenCV 종선 트래킹 및 데이터 가공 진행 중 구역입니다.")

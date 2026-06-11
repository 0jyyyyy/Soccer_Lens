import streamlit as st

# 1. 페이지 기본 설정 및 사이드바 (전역 설정)
st.set_page_config(page_title="Soccer Lens", page_icon="⚽🔭", layout="wide")

st.sidebar.title("⚽🔭 Soccer Lens")
st.sidebar.markdown("---")

# 글로벌 리그/시즌 필터(나중에 Next.js 로 이동할때 될 구성)
selected_season = st.sidebar.selectbox("📅 Select Season", ["2025/26","2026/27"])
if selected_season =="2025/26":
  selected_league = st.sidebar.selectbox(
    "🌍 Select Leagues",
    ["2026 World Cup","Premier League", "LaLiga","Bundesliga","Serie A", "Ligue 1", "K-League 1", "K-League 2"]
)
else:
  selected_league = st.sidebar.selectbox(
    "🌍 Select Leagues",
    ["Premier League", "LaLiga","Bundesliga","Serie A", "Ligue 1", "K-League 1", "K-League 2"]
)
# =====================================================
# Tactics lab 연동을 위한 세션 상태(session_state) 저장
#======================================================
st.session_state.selected_season = selected_season
st.session_state.selected_league = selected_league

#======================================================
# 중앙 메인화면(대시보드 틀)
#======================================================
st.title("⚽🔭 Soccer Lens HUB")
st.sidebar.markdown("---")

st.info(f'season: {st.session_state.selected_season} | league: {st.session_state.selected_league}')
st.success("💡 테스트 방법: 왼쪽 사이드바에서 값을 바꾼 뒤, 서브 메뉴인 `1_Tactics_Lab` 페이지로 이동해 보세요!")
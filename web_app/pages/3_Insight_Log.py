import streamlit as st

# 페이지 기본 설정
st. set_page_config(page_title="Soccer Lens - Insight_Log", page_icon="✍️", layout="wide")

# 블로그 상세페이지 변환을 위한 상태(Session State) 초기화
if "viewing_post" not in st.session_state:
  st.session_state.viewing_post = None # None 이면 목록 화면, 글 번호가 들어가면 상세화면

# 상단 헤더 섹션
st.title("✍️ Insight_Log (전술 칼럼)")
st.caption("축구의 전술적 개념과 포지션의 객관적 정의를 기록하는 공간입니다.")
st.markdown("---")

# 탭 구조 생성 (전술사 / 포지션 / 감독 가이드)
tab_position, tab_tactics, tab_manager = st.tabs(["🏃 포지션 사전", "📜 시대별 전술사", "👔 감독 연구소"])

with tab_position:
    # 1. 상단: 블로그 글 카드 목록 (언제나 위에 고정 노출)
    st.markdown("### 📰 전술 포지션 리포트 목록")
    st.write("원하는 리포트의 [📄 읽기] 버튼을 누르면 하단에 본문이 표시됩니다.")
    
    # 카드 배치를 위한 grid(열) 구성
    card_col1, card_col2, card_col3 = st.columns(3)
    
    # 블로그 글 카드 1번 (메짤라)
    with card_col1:
        box1 = st.container(border=True)
        box1.markdown("#### 🏃 #01. '메짤라(Mezzala)' 정밀 해부")
        box1.caption("🗓️ 2026.06.11 | 🏷️ 하프스페이스")
        if box1.button("📄 리포트 읽기", key="btn_mezzala"):
            st.session_state.viewing_post = "mezzala"
            
    # 블로그 글 카드 2번 (피보테)
    with card_col2:
        box2 = st.container(border=True)
        box2.markdown("#### 🛡️ #02. 전술적 회전축, '피보테(Pivote)'")
        box2.caption("🗓️ 2026.06.11 | 🏷️ 포백보호")
        if box2.button("📄 리포트 읽기", key="btn_pivote"):
            st.session_state.viewing_post = "pivote"
            
    # 블로그 글 카드 3번 (라볼피아나 - 준비중)
    with card_col3:
        box3 = st.container(border=True)
        box3.markdown("#### 🪄 #03. 변형 삼백, '라볼피아나'")
        box3.caption("🗓️ 업로드 예정 | 🏷️ 후방빌드업")
        box3.button("🔒 준비 중", key="btn_volpiana", disabled=True)
        
    st.markdown("---") # 목록과 본문을 구별하는 굵은 구분선
    
    
    # 2. 하단: 독자가 선택한 본문 내용이 채워지는 구역
    if st.session_state.viewing_post is None:
        # 아무것도 클릭하지 않은 초기 상태
        st.info("💡 위의 목록에서 읽고 싶은 전술 리포트를 선택해 주세요.")
        
    # 유저가 1번(메짤라) 카드를 클릭했을 때 목록 아래에 나타날 본문
    elif st.session_state.viewing_post == "mezzala":
        st.markdown("## 📜 [본문] 하프 스페이스의 지배자, 메짤라(Mezzala)")
        st.caption("작성자: 사커 렌즈 분석팀")
        
        # 1단계: 개념 및 기원
        st.markdown("### 🏛️ 1. 개념 및 기원")
        st.write("이탈리아어로 '반쪽 날개'를 의미하며, 현대 축구에서는 하프 스페이스를 파괴하는 공격형 미드필더입니다.")
        
        # 2단계: 전술판 움직임
        st.markdown("### ♟️ 2. 전술판 움직임 메커니즘")
        st.error("🎬 [전술판 애니메이션] 바둑알 움직임 영상 자리")
        
        # 3단계: 대표적인 선수 설명
        st.markdown("### 🏃 3. 대표적 선수별 플레이 스타일")
        st.write("* **케빈 더 브라우너:** 직선적 킬패스 및 얼리 크로스 마스터")
        st.write("* **베르나르두 실바:** 좁은 공간 볼 키핑 및 탈압박 운반형 크랙")
        
        # 4단계: 실제 경기 영상
        st.markdown("### 🎥 4. 실제 경기 매칭 분석 (Visual Evidence)")
        st.video("https://www.youtube.com/watch?v=J3g8M9bUjSg")
        
        # 본문 맨 아래에 닫기 버튼 배치
        if st.button("❌ 본문 닫기", key="close_mezzala"):
            st.session_state.viewing_post = None
            st.rerun()

    # 유저가 2번(피보테) 카드를 클릭했을 때 목록 아래에 나타날 본문
    elif st.session_state.viewing_post == "pivote":
        st.markdown("## 📜 [본문] 팀의 후방 척추이자 전술적 회전축, 피보테(Pivote)")
        st.caption("작성자: 사커 렌즈 분석팀")
        st.write("스페인어로 '축'을 뜻하며, 포백 보호와 빌드업의 기점 역할을 동시에 수행합니다.")
        st.error("🎥 [로드리/부스케츠 플레이 분석 영상] 들어올 자리")
        
        if st.button("❌ 본문 닫기", key="close_pivote"):
            st.session_state.viewing_post = None
            st.rerun()

#---------------------------------------------------------
# 기타 탭 (기본 틀 유지)
#---------------------------------------------------------
with tab_tactics:
    st.write("시대별 포메이션 변천사 공간입니다.")

with tab_manager:
    st.write("특정 감독들의 전술 시스템 연구 공간입니다.")
     
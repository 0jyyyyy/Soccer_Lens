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
        st.markdown("## 📜 하프 스페이스의 지배자, 메짤라(Mezzala)")
        
        # 1단계: 개념 및 기원
        st.markdown("# 🏛️ 1. 메짤라란?")
        st.markdown("""
                    - ‘**절반**’을 뜻하는 이탈리아어 **‘메조(Mezzo)’** 와 **‘날개’** 를 뜻하는 **‘알라(Ala)’** 의 합성어이다.  \n
                      직역하면 ‘**반쪽 날개**’라는 뜻으로, 윙어와 중앙 미드필더의 역할을 동시에 수행하는 선수나 포지션을 지칭한다.  \n
                      이같은 역할때문에 영미권에서 **하프 윙(Half Wing)** 으로 불린다.

                    ## 역사적 기원
                    - 1930년대 유행했던 W-M 포메이션 시절,
                    이탈리아에서 공격진의 윙어와 포워드를 2선에서 지원하던 미드필더들을 ‘**메짤라**’라고 부르기 시작한게 기원이다.

                    ## 현대축구에서의 의미
                    - 현대 축구에서 메짤라는 4-3-3 포메이션 3명의 미드필더 중 중앙 미드필더를 제외한 좌우에 위치하는 선수들을 일컫는다.
                    ### 역할
                    - 현대축구에서는 하프스페이스 공략의 중요성이 대두되기 시작했다. \n
                      이에 따라 중앙에만 머물지 않고 측면으로 빠져나가 수적우위를 가져오거나, \n
                      드리블 돌파로 하프스페이스 후방을 직접 타격하거나 패스를 주는 역할을 담당한다. \n
                    - 즉, 메짤라는 과거의 윙어처럼 단순히 측면에서 크로스만 올리는 역할이 아닌, \n
                      공격, 패스, 움직임을 통한 수적우위 등 다재다능함이 요구되는 역할이 됐다.
                    """)
        # 2단계: 전술판 움직임
        st.markdown("# ♟️ 2. 전술판 움직임 메커니즘")
        st.error("🎬 [전술판 애니메이션] 바둑알 움직임 영상 자리")
        
        # 3단계: 대표적인 선수 설명
        st.markdown("# 🏃 3. 대표적 선수별 플레이 스타일")
        st.write("## 중앙 지향적 - 이니에스타")
        st.write("## 측면 지향적 - 케빈 데브라이너")
        
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
     
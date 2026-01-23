import pandas as pd

df = pd.read_csv('https://www.football-data.co.uk/mmz4281/2526/F1.csv')
df.head()
# 현재 기준(2026-01-10) Ligue1의 1위는 Lens 입니다.
# Lens의 홈에서의 슈팅효율, 원정에서의 슈팅효율을 구하자
# 일단 먼저 Lens의 정보만 뽑아오는 과정
# df_teams로 2위팀 Paris SG, 꼴찌팀인 Metz
# df_teams = df[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HS','AS','HST','AST']] # 홈, 원정, 홈 득점, 원정 득점, 승무패, 홈 슈팅, 원정 슈팅, 홈 유효슈팅, 원정 유효슈팅

def get_team_stats(df, team_name):
  # 홈경기와 원정경기 필터링
  home = df[df['HomeTeam'] == team_name]
  away = df[df['AwayTeam'] == team_name]
  
  # 주요 수치 합산
  total_shots = home['HS'].sum() + away['AS'].sum()
  total_shots_target = home['HST'].sum() + away['AST'].sum()
  total_goals = home['FTHG'].sum() + away['FTAG'].sum()
  # 계산 (분모가 0일 경우를 대비해서 처리)
  shots_acc = (total_shots_target / total_shots * 100) if total_shots > 0 else 0
  target_eff = (total_goals / total_shots_target * 100) if total_shots_target > 0 else 0
  total_eff = (total_goals / total_shots * 100) if total_shots > 0 else 0
  
  # 세트피스 (코너킥)
  total_corners = home['HC'].sum() + away['AC'].sum()
  # 코너킥 대비 득점 효율
  corner_eff = (total_goals / total_corners *100) if total_corners > 0 else 0


  # print(f'[{team_name} 분석 결과]')
  # print(f'전체슈팅: {total_shots} | 유효슈팅: {total_shots_target} | 골: {total_goals}')
  # print(f'유효슈팅비율: {shots_acc:.2f}% | 유효슈팅대비득점: {target_eff:.2f}% | 전체슈팅대비득점: {total_eff:.2f}%')
  # print("-"*30)

  # 시각화용 데이터 반환
  return [
    #------공격 지표--------
    {'Team': team_name, 'Metric': 'Shot Accuracy', 'Value (%)': shots_acc},
    {'Team': team_name, 'Metric': 'Finish Effect', 'Value (%)': target_eff},
    {'Team': team_name, 'Metric': 'Goal Effect', 'Value (%)': total_eff},
    {'Team': team_name, 'Metric': 'Corner Effect', 'Value (%)':corner_eff}
  ]


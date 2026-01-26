import pandas as pd

df = pd.read_csv('https://www.football-data.co.uk/mmz4281/2526/F1.csv')

def get_team_stats(df, team_name):
  # 홈경기와 원정경기 필터링
  home = df[df['HomeTeam'] == team_name]
  away = df[df['AwayTeam'] == team_name]
  
  # 1. 공격 효율
  total_shots = home['HS'].sum() + away['AS'].sum()
  total_shots_target = home['HST'].sum() + away['AST'].sum()
  total_goals = home['FTHG'].sum() + away['FTAG'].sum()
  # 계산 (분모가 0일 경우를 대비해서 처리)
  shots_acc = (total_shots_target / total_shots * 100) if total_shots > 0 else 0 
  target_eff = (total_goals / total_shots_target * 100) if total_shots_target > 0 else 0
  total_eff = (total_goals / total_shots * 100) if total_shots > 0 else 0
  
  # 2. 세트피스 (코너킥)
  total_corners = home['HC'].sum() + away['AC'].sum()
  # 코너킥 대비 득점 효율
  corner_eff = (total_goals / total_corners *100) if total_corners > 0 else 0

  # 3. 시간대별 강점
  # 3_1 전반전 득점 합산
  first_half_goals = home['HTHG'].sum() + away['HTAG'].sum()
  # 3_2 후반전 득점 = 전체 득점 - 전반전 득점
  second_half_goals = total_goals - first_half_goals
  # 3_3 전반전 득점 비중 (%)
  first_half_goals_ratio = (first_half_goals / total_goals * 100) if total_goals > 0 else 0
  
  # 4. 수비 및 경고의 상관관계
  allow_shots_target = home['AST'].sum() + away['HST'].sum() # 홈일때 어웨이팀의 유효슈팅 허용을, 어웨이일때 홈팀의 유효슈팅 허용
  conceded_goals = home['FTAG'].sum() + away['HTAG'].sum() # 홈일때 어웨이팀의 실점, 어웨이일때 홈팀의 실점
  total_fouls = home['HF'].sum() + away['AF'].sum() # 파울
  total_yellows = home['HY'].sum() + away['AY'].sum() # 경고
  total_reds = home['HR'].sum() + away['AR'].sum() # 퇴장
  
  # print(f'[{team_name} 분석 결과]')
  # print(f'전체슈팅: {total_shots} | 유효슈팅: {total_shots_target} | 골: {total_goals}')
  # print(f'유효슈팅비율: {shots_acc:.2f}% | 유효슈팅대비득점: {target_eff:.2f}% | 전체슈팅대비득점: {total_eff:.2f}%')
  # print("-"*30)

  # 시각화용 데이터 반환
  return [
    #------공격 지표--------
    {'Team': team_name, 'Metric': '유효슈팅', 'Value (%)': shots_acc},
    {'Team': team_name, 'Metric': '유효슈팅대비득점', 'Value (%)': target_eff},
    {'Team': team_name, 'Metric': '전체슈팅대비득점', 'Value (%)': total_eff},
    {'Team': team_name, 'Metric': '코너킥대비득점', 'Value (%)':corner_eff},
    {'Team': team_name, 'Metric': '전반전득점', 'Value (%)':first_half_goals_ratio}
  ]


import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
from bs4 import BeautifulSoup

# requests.get() 으로 url 정보 요청하기
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36'}
player_list = []

for i in range(1, 21):
  url = f'https://www.transfermarkt.co.kr/spieler-statistik/wertvollstespieler/marktwertetop?ajax=yw1&page={i}'
  res = requests.get(url, headers=headers)

  # print(res.status_code) # 200 잘 나오는 거 확인
  soup = BeautifulSoup(res.content,'html.parser')

 
  player_info = soup.find_all('tr', class_=['odd','even'])
  for info in player_info:
    player = info.find_all('td')
    player[0]
    number = player[0].text
    name = player[3].text
    position = player[4].text
    age = player[5].text
    nationality = player[6].img['alt']
    team = player[7].img['alt']
    value = player[8].text.strip()
    value = value[1:-1] # '€숫자m' 일때 €의 인덱스는 [1]이고 m은 [-1]이니 이걸 제외하고 저장하는것이다.

    player_list.append([number,name,position,age,nationality,team,value]) 

df_player = pd.DataFrame(player_list, columns=['number','name','position','age','nationality','team','value'])

df_player.to_csv('world_tva.csv', index=False)
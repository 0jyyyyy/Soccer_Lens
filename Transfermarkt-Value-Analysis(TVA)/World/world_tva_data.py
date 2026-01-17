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
number = []
name = []
position = []
age = []
nationality = []
team = []
value = []

for i in range(1, 5):
  url = f'https://www.transfermarkt.co.kr/spieler-statistik/wertvollstespieler/marktwertetop?ajax=yw1&page={i}'
  res = requests.get(url, headers=headers)

  print(res.status_code) # 200 잘 나오는 거 확인
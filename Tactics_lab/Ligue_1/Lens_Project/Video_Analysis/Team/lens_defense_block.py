import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('yolov8s.pt')

# 2. 영상 설정(수비 영상 경로로 수정)
video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\second\defense\defense_block.mp4'
output_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\defense_block.mp4'

# output_path 가 없을 경우, 자동으로 폴더 생성
output_folder = os.path.dirname(output_path)
if not os.path.exists(output_folder):
  os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
  print("🚨 영상을 찾을 수 없습니다. 경로를 확인해주세요!")
  exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print('모듈 4: 수비 블록 분석 시작')

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # 사람(0)만 탐지
  results = model(frame, classes=[0], conf=0.2)

  lens_players= []
  rennes_players = []

  for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # 그라운드 밖(관중석) 제외
    if y2 < (height * 0.15) or y2 > (height * 0.95):
      continue

    w = x2 - x1
    h = y2 - y1
    shirt_img = frame[int(y1 + h*0.2) : int(y1 + h*0.5), int(x1 + w*0.3): int(x2 - w*0.3)]
    if shirt_img.size == 0: continue

    hsv_shirt = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2HSV)

    # [팀 A: 랑스] 노란색/주황색 기준
    lower_lens = np.array([15, 100, 100])
    upper_lens = np.array([35, 255, 255])
    mask_lens = cv2.inRange(hsv_shirt, lower_lens, upper_lens)

    # [팀 B: 스타드 렌] 검은색 유니폼
    lower_rennes = np.array([0, 0, 0])    
    upper_rennes = np.array([180, 255, 55])
    mask_rennes = cv2.inRange(hsv_shirt, lower_rennes, upper_rennes)

    cx, cy = int((x1 + x2) / 2), int(y2)

    if cv2.countNonZero(mask_lens) > 15:
      lens_players.append((cx, cy))
      cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1) # 랑스 빨간점
    elif cv2.countNonZero(mask_rennes) > 15:
      rennes_players.append((cx, cy))
      cv2.circle(frame, (cx, cy), 8, (0, 0, 0), -1) # 렌 검은점
      cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2) # 테두리 강조

  # 수비 블록(두줄 수비) 시각화
  # 최소 6명 이상 화면에 잡혔을 때 두줄 (수비 + 미드필더)로 나눕니다.
  if len(lens_players) >= 6:
    # 1. 선수들을 x축(가로 방향) 기준으로 정렬
    lens_players.sort(key=lambda p: p[0])
    
    # 2. 선수를 반으로 갈라서 두 개 의 라인으로 분리
    mid_idx = len(lens_players) // 2
    line1 = lens_players[:mid_idx] # 왼쪽에 있는 그룹 (예: 수비라인)
    line2 = lens_players[mid_idx:] # 오른쪽에 있는 그룹 (예: 미드필더 라인)

    # 3. 각 라인 안에서는 y축(세로 방향) 기준으로 다시 정렬 
    # 그래야 선이 꼬이지 않고 예쁘게 이어진다.
    line1.sort(key=lambda p: p[1])
    line2.sort(key=lambda p: p[1])

    # 4. 초록색의 굵은 두 줄 수비 라인 긋기
    pts1 = np.array(line1, np.int32)
    pts2 = np.array(line2, np.int32)
    cv2.polylines(frame, [pts1], False, (0, 255, 0), 3) # False 로 해야 다각형으로 생성되지않고 선으만 표시됨
    cv2.polylines(frame, [pts2], False, (0, 255, 0), 3)

    # 5. (핵심) 두 라인 사이의 '간격(Pocket Space)' 을 옅은 초록색으로 칠해주기
    # line1 과 역순으로 뒤집은 line2를 연결해서 하나의 닫힌 영역을 만든다.
    pocket_pts = np.array(line1 + line2[::-1], np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pocket_pts], (0, 255, 0))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame) # 15% 투명도
  out.write(frame)
  cv2.imshow('Module 4: Lens vs Rennes defensive block', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
out.release()
cv2.destroyAllWindows()
print('분석 완료!')
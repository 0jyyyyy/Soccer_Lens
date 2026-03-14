import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('yolov8s.pt')

# 2. 영상 설정 (수비 영상 경로로 수정)
video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\first\defense\defense_foul_1.mp4'
output_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\defense_foul_1.mp4'

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

print('모듈 3: 수비 간격 분석 시작')

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # 사람(0)만 탐지
  results = model(frame, classes=[0], conf=0.2)

  lens_players = []
  rennes_players = []

  for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # 그라운드 밖(관중석) 제외
    if y2 < (height * 0.15) or y2 > (height * 0.95):
      continue

    w = x2 - x1
    h = y2 - y1
    shirt_ing = frame[int(y1 + h*0.2) : int(y1 + h*0.5), int(x1 + w*0.3) : int(x2 - w*0.3)]
    if shirt_ing.size == 0: continue

    hsv_shirt = cv2.cvtColor(shirt_ing, cv2.COLOR_BGR2HSV)

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
  # 수비 라인 간격(gap) 분석
  if len(lens_players) >= 2:
    # 선수들을 x좌표(왼쪽에서 오른쪽) 순서로 정렬하여 수비 라인 구축
    lens_players.sort(key=lambda p: p[0])   

    for i in range(len(lens_players) - 1):
      pt1 = lens_players[i]
      pt2 = lens_players[i+1]
      dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5

      # 간격이 너무 넓게 벌어지면 (역습 위기 상황)
      if dist > 350: # 찢어진 간격 기준 픽셀
        # 붉은색 굵은 선으로 위험 공간 표시
        cv2.line(frame, pt1, pt2, (0,0,255),4)

        # 그 공간 한가운데에 경고 텍스트 띄우기
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2)
        cv2.putText(frame, 'DANGER GAP', (mid_x - 60, mid_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
      else:
        # 간격이 촘촘하면 안전한 녹색 선
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
  out.write(frame)
  cv2.imshow('Module 3: Lens vs Rennes defensive gap', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
out.release()
cv2.destroyAllWindows()
print('분석 완료!')
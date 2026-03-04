import cv2
import numpy as np
from ultralytics import YOLO

# 1. 모델 로드 (공 인식을 위해 yolov8s.pt)
model = YOLO('yolov8s.pt')

# 2. 영상 설정
video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\first\press\team_attack_1.mp4' # 공격 전개 영상
output_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\team_attack_1.mp4'

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,(width,height))

# 공의 궤적을 저장할 리스트
ball_trail = []

print("lens_attack 분석 시작...")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # 사람(0)과 공(32) 탐지
  results = model(frame, classes=[0, 32], conf=0.15)

  lens_players = []

  for box in results[0].boxes:
    x1,y1,x2,y2= map(int,box.xyxy[0])
    class_id = int(box.cls[0])

    # [공(ball)인 경우]
    if class_id == 32:
      ball_cx, ball_cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
      ball_trail.append((ball_cx, ball_cy))
      # 공 위치 표시
      cv2.circle(frame, (ball_cx, ball_cy), 8, (255, 255, 255), -1)
      cv2.circle(frame, (ball_cx, ball_cy), 8, (0, 0, 0), 2)

    # [사람(Person)인 경우 = 랑스 선수만 필터링]
    if class_id == 0:
      shirt_img = frame[y1:int(y1 + (y2-y1)*0.5), x1:x2]
      if shirt_img.size == 0: continue

      hsv_shirt = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2HSV)
      lower_lens = np.array([10, 50, 50])
      upper_lens = np.array([40, 255, 255])
      mask_lens = cv2.inRange(hsv_shirt, lower_lens, upper_lens)

      # 랑스 선수로 판별 되면 리스트에 추가
      if cv2.countNonZero(mask_lens) > 20:
        cx, cy = int((x1 + x2) / 2), int(y2)
        lens_players.append((cx, cy))
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1) # 발밑에 빨간 점
        

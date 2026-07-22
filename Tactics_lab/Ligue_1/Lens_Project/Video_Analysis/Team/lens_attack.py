import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. 모델 로드 (공 인식을 위해 yolov8s.pt)
model = YOLO('yolov8s.pt')

# 2. 영상 설정
video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\first\attack\team_attack_1.mp4' # 공격 전개 영상
output_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\team_attack_1.mp4'

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
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,(width,height))

print("모듈 1: 공격패턴 분석 시작...")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # 사람(0)만 탐지
  results = model(frame, classes=[0], conf=0.2)

  lens_players = []

  for box in results[0].boxes:
    x1,y1,x2,y2= map(int,box.xyxy[0])

    # 화면 맨 위(관중석), 맨 아래는 제외
    if y2 < (height * 0.15) or y2 > (height * 0.95):
      continue

    # ★ 핵심 필터링: 선수의 '가슴 정중앙' 영역만 타이트하게 잘라냄
    w = x2 - x1
    h = y2 - y1
    shirt_img = frame[int(y1 + h*0.2) : int(y1 + h*0.5), int(x1 + w*0.3) : int(x2 - w*0.3)]
        
    if shirt_img.size == 0: continue

    hsv_shirt = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2HSV)
        
    # ★ 노란색 기준을 엄청나게 엄격하게 상향 (채도 100 이상, 명도 100 이상)
    lower_lens = np.array([15, 100, 100])
    upper_lens = np.array([35, 255, 255])
    mask_lens = cv2.inRange(hsv_shirt, lower_lens, upper_lens)

    # 랑스 선수로 판별 (정중앙에 노란색이 확실히 있을 때만)
    if cv2.countNonZero(mask_lens) > 15:
      cx, cy = int((x1 + x2) / 2), int(y2)
      lens_players.append((cx, cy))
      # 랑스 선수 발밑에 깔끔한 빨간 원 하나만 표시
      cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1) 
      cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2) # 흰색 테두리로 강조

    # 3. 랑스의 팀 대형(공간) 시각화 - 깔끔한 테두리만!
    if len(lens_players) >= 4: # 최소 4명 이상 잡혔을 때만 블록 형성
      pts = np.array(lens_players, np.int32)
      hull = cv2.convexHull(pts)

      overlay = frame.copy()
      # 내부 색칠은 아주 연하게 
      cv2.fillPoly(overlay, [hull], (0, 165, 255))
      cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame) 
        
      # 굵고 명확한 주황색 외곽선만 그리기 (초록색 거미줄 완전 삭제)
      cv2.polylines(frame, [hull], True, (0, 165, 255), 2)

  out.write(frame)
  cv2.imshow('Lens Clean Attack Block', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
out.release()
cv2.destroyAllWindows()
print('Analysis 완')
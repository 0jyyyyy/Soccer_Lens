import cv2
import numpy as np
from ultralytics import YOLO

# 1. 모델 로드 (공 인식을 위해 yolov8s.pt)
model = YOLO('yolov8s.pt')

# 2. 영상 설정
video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\first\attack\team_attack_1.mp4' # 공격 전개 영상
output_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\team_attack_2.mp4'

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
      # 1. AI의 확신도(Confidence Score) 가져오기
      conf_score = float(box.conf[0])
      
      # 중심 좌표 계산
      ball_cx, ball_cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
      # 순간이동 방지 로직
      if len(ball_trail) > 0:
        last_x, last_y = ball_trail[-1]
        dist = ((ball_cx - last_x)**2 + (ball_cy - last_y)**2)**0.5
        # 이전 공 위치에서 너무 멀리 떨어져 있으면 가짜로 간주하고 무시!
        if dist > 150: 
          continue
      # 2. 강력한 필터링 조건 2가지 적용
      # 조건 A: AI가 30% 이상(0.3) 확실하다고 판단할 때만 인정 (깃발 같은 애매한 것 탈락)
      # 조건 B: 화면 상단 20% (관중석 영역)에 있는 공은 무조건 가짜로 간주하고 무시
      if conf_score > 0.3 and ball_cy > (height * 0.2):
          ball_trail.append((ball_cx, ball_cy))
          # 공 위치 표시
          cv2.circle(frame, (ball_cx, ball_cy), 8, (255, 255, 255), -1)
          cv2.circle(frame, (ball_cx, ball_cy), 8, (0, 0, 0), 2)

    # [사람(Person)인 경우 = 랑스 선수만 필터링]
    if class_id == 0:
      # 그라운드 출입 통제 (화면 상하단 15% 영역에 있는 사람은 무시)
      if y2 < (height * 0.15) or y2 > (height * 0.90):
        continue

      shirt_img = frame[y1:int(y1 + (y2-y1)*0.5), x1:x2]
      if shirt_img.size == 0: continue

      hsv_shirt = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2HSV)
            
      # 노란색/주황색 범위 (조금 더 엄격하게 채도 조정)
      lower_lens = np.array([10, 80, 50])  # 채도(Saturation) 하한선을 50->80으로 높임
      upper_lens = np.array([35, 255, 255])
      mask_lens = cv2.inRange(hsv_shirt, lower_lens, upper_lens)

      # 랑스 선수 판별
      if cv2.countNonZero(mask_lens) > 30: # 픽셀 조건도 20->30으로 엄격하게
        cx, cy = int((x1 + x2) / 2), int(y2)
        lens_players.append((cx, cy))
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1) # 발밑에 빨간 점
  # 랑스의 팀 대형(공간) 시각화
  if len(lens_players) >= 3:
    # 전체 대형을 감싸는 블록(Convex Hull) 그리기
    pts = np.array(lens_players, np.int32)
    hull = cv2.convexHull(pts)

    overlay = frame.copy()
    # 공격 시에는 팽창하는 느낌을 주기 위해 옅은 붉은색/주황색 사용
    cv2.fillPoly(overlay, [hull],(0, 165, 255))
    cv2.addWeighted(overlay, 0.2,frame,0.8, 0, frame) # 20% 투명도
    cv2.polylines(frame, [hull], True, (0, 165, 255), 2) # 테두리

    # 팀 내부 네트워크(선수 간 연결) 그리기
    for i in range(len(lens_players)):
      for j in range(i + 1, len(lens_players)):
        cv2.line(frame, lens_players[i],lens_players[j],(0, 255, 0), 1)

  out.write(frame)
  cv2.imshow('lens attack & formation shift', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
out.release()
cv2.destroyAllWindows()
print('Analysis 완')
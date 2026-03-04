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

  # 공의 패스 궤적 시각화 (선수들 위로 날아가는 롱패스 궤적)
  if len(ball_trail) > 1:
    # 화면에 너무 길게 남지 않도록 최근 40프레임(약 1.5초)만 유지
    if len(ball_trail) > 40:
      ball_trail.pop(0)
    ball_pts = np.array(ball_trail, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [ball_pts], isClosed=False, color=(255, 255, 255), thickness=4)

    out.write(frame)
    cv2.imshow('랑스 팀 공격 & 움직임패턴', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
out.release()
cv2.destroyAllWindows()
print('Analysis 완')
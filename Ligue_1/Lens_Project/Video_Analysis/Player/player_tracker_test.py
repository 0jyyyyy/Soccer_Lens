import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\first\thauvin\move\thauvin_movement_4.mp4'
output_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\player\thauvin_movement_4.mp4'

# 핵심 : 추적할 선수의 ID 번호
# (처음엔 아무 번호나 넣고 영상을 돌려본 뒤, 타겟 선수의 머리 위 번호를 확인하고 TARGET_ID를 변경)
TARGET_ID = 15

# 꼬리의 길이 (30프레임 = 약 1초 전까지의 궤적을 남김. 길게 남기려면 60, 90으로 세팅)
TRAIL_LENGTH = 45

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

# 선수의 궤적(X, Y 좌표)을 기록할 메모리 장부
track_history = defaultdict(lambda: [])

print(" ByteTrack start")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret: break

  # 핵심: predict 대신 track을 사용하고, tracker='bytetrack.yaml'을 킨다.
  results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)

  if results[0].boxes.id is not None:
    boxes = results[0].boxes.xywh.cpu() # x, y, width, height
    track_ids = results[0].boxes.id.int().cpu().tolist() # ByteTrack이 부여한 고유 번호들

    for box, track_id in zip(boxes, track_ids):
      x, y, w, h = box

      # 발밑 좌표 계산 (꼬리가 머리나 배가 아닌 '발끝'에 나오도록)
      cx, cy = int(x), int(y + h / 2)

      # 1. 궤적 장부에 현재 발밑 위치 기록
      track = track_history[track_id]
      track.append((cx, cy))

      # 꼬리가 너무 길어지면 옛날 기록부터 지우기
      if len(track) > TRAIL_LENGTH:
        track.pop(0)

      # 2. 타겟 선수(TARGET_ID)에게만 스포트라이트와 꼬리 부여
      if track_id == TARGET_ID:
        # [스포트라이트] 발밑에 눈에 띄는 타겟팅 원 그리기(형광 노란색)
        cv2.ellipse(frame, (cx, cy), (int(w/2), int(w/4)), 0, 0, 360, (0, 255, 255), 3)

        # [스포트라이트] 머리 위에 ID 번호 띄우기
        cv2.putText(frame, f'TARGET: {track_id}', (int(x - w/2), int(y - h/2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # [무브먼트 꼬리] 기록된 궤적을 굵은 선으로 쫙 이어주기 (강렬한 주황색)
        if len(track) > 1:
          pts = np.array(track, np.int32).reshape((-1, 1, 2))
          cv2.polylines(frame, [pts], False, (0, 165, 255), 4)
      # 다른 모든 선수들도 번호를 작게 띄워놔야 나중에 타겟 ID 찾기가 편리
      else:
        cv2.putText(frame, str(track_id), (int(x), int(y - h/2)),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
  out.write(frame)
  cv2.imshow('tracking & trail', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print('tracking done')           
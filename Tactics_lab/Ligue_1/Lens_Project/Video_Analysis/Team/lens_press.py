import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('yolov8s.pt')

# 2. 영상 설정 (압박 영상 경로로 수정하세요)
video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\first\press\team_press & defense_gap.mp4' 
output_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\high_press_result_1.mp4'

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

print("모듈 2: 전방 압박(창과 방패) 분석 시작...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 사람(0)만 탐지 (공 궤적은 깔끔함을 위해 제외)
    results = model(frame, classes=[0], conf=0.2)

    lens_players = []
    rennes_players = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 그라운드 밖(관중석, 벤치) 사람 제외
        if y2 < (height * 0.15) or y2 > (height * 0.95):
            continue

        # 상의 가슴팍 부분만 잘라내기
        w = x2 - x1
        h = y2 - y1
        shirt_img = frame[int(y1 + h*0.2) : int(y1 + h*0.5), int(x1 + w*0.3) : int(x2 - w*0.3)]
        
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

        # 팀 분류 및 좌표 저장
        cx, cy = int((x1 + x2) / 2), int(y2)
        if cv2.countNonZero(mask_lens) > 15:
            lens_players.append((cx, cy))
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1) # 랑스: 빨간점
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
        elif cv2.countNonZero(mask_rennes) > 15:
            rennes_players.append((cx, cy))
            cv2.circle(frame, (cx, cy), 8, (0, 0, 0), -1) # 렌: 파란점
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)

    # 3. 시각화 A: 랑스의 전방 압박 블록 (붉은색 거미줄 덫)
    for rennes_pt in rennes_players:
        rx, ry = rennes_pt
        pressers = []
        
        # 현재 스타드 렌 선수를 기준으로 반경 250픽셀 이내에 있는 랑스 선수 찾기
        for lens_pt in lens_players:
            lx, ly = lens_pt
            dist = ((rx - lx)**2 + (ry - ly)**2)**0.5
            if dist < 250:  # 압박 반경 (덫의 크기. 너무 넓으면 300, 좁으면 200으로 조절 가능)
                pressers.append(lens_pt)
                
        # 렌 선수 1명 주위에 랑스 선수가 3명 이상 달라붙었다면 = 완벽한 '덫' 완성!
        if len(pressers) >= 3:
            pts = np.array(pressers, np.int32)
            hull = cv2.convexHull(pts)
            
            # 1. 덫(다각형) 내부를 붉고 투명하게 칠하기 (긴장감 조성)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [hull], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame) # 30% 투명도 (눈에 확 띄게!)
            
            # 2. 덫의 테두리 (강렬한 붉은 선)
            cv2.polylines(frame, [hull], True, (0, 0, 255), 2)
            
            # 3. 갇혀버린 렌 선수(사냥감) 강조 (발밑에 노란색 타겟팅 원 그리기)
            cv2.circle(frame, (rx, ry), 15, (0, 255, 255), 3)

    out.write(frame)
    cv2.imshow('Module 2: Lens High Press vs Rennes', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print('분석 완료!')
import cv2
import numpy as np
import os
from ultralytics import YOLO

# 프로그램 시작 및 사용자 입력(CLI)

print('=' * 50)
print(" ⚽ Soccer Lens: Team_Vidoe_Analysis_Engine ⚽")
print('=' * 50)
print(' [1] 공격 패턴')
print(' [2] 전방 압박')
print(' [3] 역습 방어(Danger Gap)')
print(' [4] 수비 블럭')
print('=' * 50)

# 1. 모드 입력받기
while True:
  mode_input = input('▶ 실행할 분석 모드 번호를 입력하세요 (1~4): ')
  if mode_input in ['1','2','3','4']:
    MODE = int(mode_input)
    break
  print('1에서 4 사이의 숫자를 입력해주세요!')

# 2. 영상 경로 입력받기 (드래그 앤 드롭 시 생기는 따옴표 자동 제거)
video_path = input('▶ 분석할 영상의 경로를 입력(또는 드래그)하세요: ').strip('\"\'')
# 사용자가 실수로 .mp4를 빼먹었다면 알아서 붙여주기!
if not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
    video_path += '.mp4'


# 3. 결과물을 저장할 폴더도 입력받기 (엔터 치면 기본값 자동 세팅)

print("  ('엔터'를 치시면 원본 영상이 있는 곳에 output 폴더를 자동 생성합니다!)")
output_dir_input = input("▶ 결과물을 저장할 폴더 경로를 입력(또는 드래그)하세요: ").strip('\"\'')

# 만약 사용자가 아무것도 입력 안 하고 엔터만 쳤다면? (기본 경로 자동 세팅)
if not output_dir_input:
    output_dir_input = os.path.join(os.path.dirname(video_path), 'output')

# 폴더가 없으면 생성하고, 최종 저장 경로 완성하기
os.makedirs(output_dir_input, exist_ok=True)
video_name = os.path.basename(video_path)
output_path = os.path.join(output_dir_input, f'mode{MODE}_{video_name}')

# ======================================================================
# ⚙️ 팀 색상 설정 (유니폼이 바뀔 때 여기만 수정하세요!)
# ======================================================================
# 🎯 타겟 팀 (현재 세팅: 노란색/주황색)
TARGET_LOWER_HSV = np.array([15, 100, 100])
TARGET_UPPER_HSV = np.array([35, 255, 255])
TARGET_POINT_COLOR = (0, 0, 255)    # 발밑 점 색상 (빨강)
TARGET_DRAW_COLOR = (0, 165, 255)   # 다각형/선 색상 (주황) - 모드별로 덮어쓰기 됨

# ⚔️ 상대 팀 (현재 세팅: 검은색)
OPPONENT_LOWER_HSV = np.array([0, 0, 0])
OPPONENT_UPPER_HSV = np.array([180, 255, 55])
OPPONENT_POINT_COLOR = (0, 0, 0)    # 발밑 점 색상 (검정)

# ======================================================================
# 🚀 메인 분석 엔진 가동
# ======================================================================
print(f"\n🔥 Team_Vidoe_Analysis_Engine - MODE {MODE} 분석을 시작합니다...")

model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("🚨 영상을 찾을 수 없습니다. 경로를 다시 확인해주세요!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 사람 탐지
    results = model(frame, classes=[0], conf=0.2)
    
    target_players = []
    opponent_players = []

    # 피아 식별 (HSV 필터링)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if y2 < (height * 0.15) or y2 > (height * 0.95): continue # 관중석, 트랙 제외

        w, h = x2 - x1, y2 - y1
        shirt_img = frame[int(y1 + h*0.2) : int(y1 + h*0.5), int(x1 + w*0.3) : int(x2 - w*0.3)]
        if shirt_img.size == 0: continue

        hsv_shirt = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2HSV)
        
        mask_target = cv2.inRange(hsv_shirt, TARGET_LOWER_HSV, TARGET_UPPER_HSV)
        mask_opponent = cv2.inRange(hsv_shirt, OPPONENT_LOWER_HSV, OPPONENT_UPPER_HSV)

        cx, cy = int((x1 + x2) / 2), int(y2)
        
        # 타겟 팀 분류
        if cv2.countNonZero(mask_target) > 15:
            target_players.append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, TARGET_POINT_COLOR, -1) 
        
        # 상대 팀 분류
        elif cv2.countNonZero(mask_opponent) > 15:
            opponent_players.append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, OPPONENT_POINT_COLOR, -1)
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2) # 테두리 강조

    # ======================================================================
    # 🎨 선택된 모드에 따른 전술 시각화 (Tactical Drawing)
    # ======================================================================

    # [MODE 1] 공격 패턴
    if MODE == 1 and len(target_players) >= 4:
        hull = cv2.convexHull(np.array(target_players, np.int32))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull], TARGET_DRAW_COLOR)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.polylines(frame, [hull], True, TARGET_DRAW_COLOR, 2)

    # [MODE 2] 전방 압박
    elif MODE == 2:
        for rx, ry in opponent_players:
            pressers = [pt for pt in target_players if ((rx - pt[0])**2 + (ry - pt[1])**2)**0.5 < 250]
            if len(pressers) >= 3:
                hull = cv2.convexHull(np.array(pressers, np.int32))
                overlay = frame.copy()
                cv2.fillPoly(overlay, [hull], (0, 0, 255)) # 압박은 강렬한 빨강
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [hull], True, (0, 0, 255), 2)
                cv2.circle(frame, (rx, ry), 15, (0, 255, 255), 3) # 타겟팅 과녁

    # [MODE 3] 역습 방어(Danger Gap)
    elif MODE == 3 and len(target_players) >= 2:
        target_players.sort(key=lambda p: p[0])
        for i in range(len(target_players) - 1):
            pt1, pt2 = target_players[i], target_players[i+1]
            if ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5 > 350:
                cv2.line(frame, pt1, pt2, (0, 0, 255), 4) # 찢어진 간격 (빨강)
                cv2.putText(frame, "DANGER GAP!", (int((pt1[0]+pt2[0])/2) - 60, int((pt1[1]+pt2[1])/2) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2) # 안전한 간격 (초록)

    # [MODE 4] 수비 블럭
    elif MODE == 4 and len(target_players) >= 6:
        target_players.sort(key=lambda p: p[0])
        mid_idx = len(target_players) // 2
        line1, line2 = target_players[:mid_idx], target_players[mid_idx:]
        line1.sort(key=lambda p: p[1])
        line2.sort(key=lambda p: p[1])
        
        pts1, pts2 = np.array(line1, np.int32), np.array(line2, np.int32)
        cv2.polylines(frame, [pts1], False, (0, 255, 0), 3) # 수비 라인 (초록)
        cv2.polylines(frame, [pts2], False, (0, 255, 0), 3) # 미드필더 라인 (초록)
        
        pocket_pts = np.array(line1 + line2[::-1], np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pocket_pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    out.write(frame)
    cv2.imshow(f'Team_Vidoe_Analysis_Engine', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n 분석 완료! 결과물이 다음 경로에 저장되었습니다:\n{output_path}")
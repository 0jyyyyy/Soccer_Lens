import cv2
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('yolov8n.pt')

# 2. 영상 경로 설정 (테스트할 4개 영상 중 하나씩 입력하세요)
video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\first\press\team_attack_1.mp4'  # 예: 전반 팀 공격패턴
video_path_2 = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\first\press\team_press & defense_gap.mp4'  # 예: 전반 전방압박 & 수비간격
video_path_3 = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\second\defense\defense_gap_3.mp4'  # 예: 후반 수비간격
output_path = 'team_attack_1.mp4'
output_path_2 = 'team_press & defense_gap.mp4'
output_path_3 = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\defense_gap_3.mp4'

cap = cv2.VideoCapture(video_path_3)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(output_path_3, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"[{video_path_3}] 전술 분석을 시작합니다...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # 사람만 탐지 (classes=0)
    results = model(frame, classes=0, conf=0.3)
    
    centers = []
    
    # 선수들의 발밑 좌표 추출
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int(y2) # 발밑 중앙
        centers.append((cx, cy))
        
        # 선수 위치에 점 찍기
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1) # 노란색 점

    # 선수들 간의 네트워크(간격) 선 긋기
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pt1 = centers[i]
            pt2 = centers[j]
            
            # 두 선수 사이의 거리 계산
            distance = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
            
            # 거리에 따라 선 색상과 굵기 다르게 표현 (수치는 영상에 맞춰 조절 가능)
            if distance < 150:
                # 매우 촘촘함 (강한 압박 or 블록 형성)
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2) # 굵은 녹색 선
            elif 150 <= distance < 300:
                # 적정 간격 유지
                cv2.line(frame, pt1, pt2, (255, 255, 0), 1) # 얇은 옥색/노란색 선
            elif 300 <= distance < 450:
                # 간격이 벌어짐 (위험 노출)
                cv2.line(frame, pt1, pt2, (0, 0, 255), 1) # 얇은 빨간색 선

    out.write(frame)
    
    # 화면 출력 (작업 과정 모니터링)
    display_frame = cv2.resize(frame, (960, 540))
    cv2.imshow('Tactical Network Analysis', display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"분석 완료! 결과 영상: {output_path_3}")
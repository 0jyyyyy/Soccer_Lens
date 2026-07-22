import cv2
from ultralytics import YOLO

# 1. YOLOv8 모델 로드(처음 실행 시 자동으로 가벼운 모델인 yolov8n.pt를 다운로드)
model = YOLO('yolov8n.pt')

# 2. 비디오 입력 및 출력 설정
# 여기에 분류해놨던 경기 영상중 하나의 경로를 입력한다.
video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\first\sarr\bulidup\sarr and thauvin.mp4' 
cap = cv2.VideoCapture(video_path)

# 영상 저장을 위한 설정 (원본 영상의 크기와 FPS를 그대로 가져옵니다)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 결과를 저장할 파일 이름 설정
output_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\sarr and thauvin.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("영상 분석을 시작합니다... (종료하려면 영상 창에서 'q'를 누르세요)")
while cap.isOpened():
  ret, frame = cap.read()
  if not ret: # 영상이 끝나면 루프 종료
    break

  # 3. 객체 탐지
  # classes=0 (사람만 탐지), conf=0.3 (신뢰도 30% 이상인 것만 표시)
  results = model(frame, classes=0, conf=0.3)

  # 4. 화면에 박스 그리기 (YOLOv8 자체 기능으로 아주 쉽게 그려진다.)
  annotated_frame = results[0].plot()

  # 5. 결과 영상 저장 및 화면 출력
  out.write(annotated_frame)

  # 화면에 크기를 살짝 줄여서 보여주기 (노트북 화면에 맞게)
  display_frame = cv2.resize(annotated_frame, (960, 540))
  cv2.imshow('토뱅, 사르 연계 (테스트)', display_frame)

  # 'q' 키를 누르면 중간에 강제 종료
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
# 메모리 해제 및 창 닫기
cap.release()
out.release()
cv2.destroyAllWindows()
print(f'분석 완료 결과 영상이 {output_path} 로 저장')
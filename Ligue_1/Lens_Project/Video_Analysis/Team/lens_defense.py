import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('yolov8s.pt')

# 2. 영상 설정 (수비 영상 경로로 수정)
video_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\second\defense\defense_gap.mp4'
output_path = r'C:\Users\ojy05\Videos\LENS_video\STADE REN\output\defense_gap.mp4'
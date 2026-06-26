# -*- coding: utf-8 -*-
"""
손 모양(제스처) 인식 프로그램 (각도 기반 개선 버전)
------------------------------------------
손가락의 펴짐/접힘을 단순 좌표 비교가 아닌, 
3D 관절 사이의 '각도(Angle)'를 계산하여 판별합니다.
이로 인해 손등, 손바닥, 손의 회전과 관계없이 안정적으로 제스처를 인식합니다.
"""

import os
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ---------------------------------------------------------------
# 0. 모델 파일 준비
# ---------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print("손 인식 모델 파일을 처음 한 번 다운로드합니다...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("다운로드 완료:", MODEL_PATH)


# ---------------------------------------------------------------
# 1. MediaPipe HandLandmarker 모델 로드
# ---------------------------------------------------------------
def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,                       # 최대 2개의 손 인식
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


# 관절 각도 계산을 위한 인덱스 정의 (완성된 3개 관절 세트)
# (기저 관절, 중간 관절, 손가락 끝) 순서
FINGER_JOINTS = {
    "thumb": (2, 3, 4),      # MCP, IP, TIP
    "index": (5, 6, 8),      # MCP, PIP, TIP (안정성을 위해 중간 마디 건너뛰고 끝과 비교)
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20)
}


def calculate_angle(p1, p2, p3):
    """
    세 점 p1, p2, p3가 이루는 3차원 공간상의 각도(도 단위)를 계산합니다.
    p2가 중심점(꼭짓점)이 됩니다.
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    # 벡터 정규화
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    dot_product = np.dot(v1, v2) / (norm_v1 * norm_v2)
    # 부동소수점 오차로 인해 -1 ~ 1 범위를 벗어나는 것 방지
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle = np.arccos(dot_product)
    return np.degrees(angle)


def get_finger_states_by_angle(landmarks):
    """
    관절의 꺾임 각도를 기준으로 손가락이 펴졌는지 판단합니다.
    일반적으로 펴진 손가락은 세 점이 이루는 각도가 160도~180도에 가깝습니다.
    """
    states = {}
    
    for name, joints in FINGER_JOINTS.items():
        p1 = landmarks[joints[0]]
        p2 = landmarks[joints[1]]
        p3 = landmarks[joints[2]]
        
        angle = calculate_angle(p1, p2, p3)
        
        # 엄지는 구조상 조금만 펴져도 각도가 나오므로 기준 완화 (150도)
        # 나머지 손가락은 완전히 펴졌을 때 대략 160도 이상
        if name == "thumb":
            states[name] = angle > 150
        else:
            states[name] = angle > 160
            
    return states


def classify_gesture(states):
    """펴진 손가락 조합을 보고 제스처 분류"""
    thumb, index, middle, ring, pinky = (
        states["thumb"], states["index"], states["middle"],
        states["ring"], states["pinky"],
    )

    if thumb and not index and not middle and not ring and not pinky:
        return "1. um"
    if not thumb and index and not middle and not ring and pinky:
        return "2. lock and lol"
    if not thumb and index and middle and not ring and not pinky:
        return "3. seisor"
    if not thumb and not index and not middle and not ring and not pinky:
        return "4. mook"
    if thumb and index and middle and ring and pinky:
        return "5. bo"

    return "wrong"


def draw_landmarks_on_frame(frame, hand_landmarks):
    """OpenCV로 직접 손 뼈대(스켈레톤)를 그려줌"""
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
        (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
        (0, 9), (9, 10), (10, 11), (11, 12),     # 중지
        (0, 13), (13, 14), (14, 15), (15, 16),   # 약지
        (0, 17), (17, 18), (18, 19), (19, 20),   # 소지
        (5, 9), (9, 13), (13, 17),               # 손바닥 라인
    ]

    for a, b in connections:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 100, 255), -1)


def main():
    ensure_model_downloaded()
    landmarker = create_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 카메라 연결/권한을 확인해주세요.")
        return

    print("프로그램 시작! 종료하려면 영상 창에서 'q'를 누르세요.")

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)  # 셀카 모드처럼 좌우 반전
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((frame_idx / fps) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        if not result.hand_landmarks:
            cv2.putText(
                frame, "no detection", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )
        else:
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                draw_landmarks_on_frame(frame, landmarks)

                label = handedness[0].category_name  # "Left" / "Right"
                
                # [개선 핵심] 방향성 타지 않는 3D 각도 기반 상태 판별
                states = get_finger_states_by_angle(landmarks)
                gesture_text = classify_gesture(states)

                # 손가락 근처에 글씨 띄우기
                text_x = int(landmarks[5].x * w)
                text_y = int(landmarks[5].y * h) - 30
                text_y = max(text_y, 30)

                display_string = f"[{label}] {gesture_text}"
                
                cv2.putText(
                    frame, display_string, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
                )

        cv2.imshow("Hand Gesture Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
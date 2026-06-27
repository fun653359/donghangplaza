# -*- coding: utf-8 -*-
"""
손모양 따라하기 게임 (메인 화면/로직)
------------------------------------------
- hand_recognize.py 의 함수(ensure_model_downloaded, create_landmarker,
  get_finger_states_by_angle, classify_gesture)를 그대로 가져와서 사용합니다.
  hand_recognize.py 파일 자체는 수정하지 않습니다.
- 대기 화면에서 키보드 1/2/3 으로 난이도(하/중/상)를 고르면 게임이 시작됩니다.
- 10라운드 동안 무작위(난이도별 패턴)로 1.jpg~10.jpg 이미지를 보여주고,
  카메라로 같은 손모양을 인식하면 O(초록), 아니면 X(빨강)를 표시합니다.
- 한 라운드는 3초간 진행되며, 시간이 지나면 맞았든 틀렸든 다음 라운드로 넘어갑니다.
- 카메라 화면(PIP)은 화면 우하단에 항상 표시됩니다.
"""

import os
import random
import time

import cv2
import pygame
import mediapipe as mp

# hand_recognize.py 안의 함수들을 그대로 가져와 재사용 (파일 자체는 수정하지 않음)
from hand_recognize import (
    ensure_model_downloaded,
    create_landmarker,
    get_finger_states_by_angle,
    classify_gesture,
)

# ---------------------------------------------------------------
# 화면/레이아웃 설정
# ---------------------------------------------------------------
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 680

# PIP(웹캠 미리보기) 크기와 위치 - 우하단
PIP_WIDTH = 320
PIP_HEIGHT = 240
PIP_MARGIN = 20
PIP_X = SCREEN_WIDTH - PIP_WIDTH - PIP_MARGIN
PIP_Y = SCREEN_HEIGHT - PIP_HEIGHT - PIP_MARGIN

ROUND_DURATION_SEC = 3.0   # 한 라운드 진행 시간
TOTAL_ROUNDS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ---------------------------------------------------------------
# 이미지 번호 <-> 손모양 매핑
# 1,2 = 가위 / 3,4 = 보 / 5,6 = 락앤롤 / 7,8 = 엄지 / 9,10 = 바위
# classify_gesture()가 돌려주는 라벨과 연결
# ---------------------------------------------------------------
GESTURE_OF_IMAGE = {
    1: "3. seisor",      # 가위
    2: "3. seisor",      # 가위
    3: "5. bo",          # 보
    4: "5. bo",          # 보
    5: "2. lock and lol",  # 락앤롤
    6: "2. lock and lol",  # 락앤롤
    7: "1. um",          # 엄지
    8: "1. um",          # 엄지
    9: "4. mook",        # 바위 (hand_recognize.py 상의 라벨은 "묵")
    10: "4. mook",       # 바위
}

# 사람이 보기 좋은 한글 이름 (디버그/표시용)
KOREAN_NAME_OF_GESTURE = {
    "1. um": "엄지",
    "2. lock and lol": "락앤롤",
    "3. seisor": "가위",
    "4. mook": "바위",
    "5. bo": "보",
    "wrong": "?",
}

IMAGE_IDS_BY_GESTURE = {
    "3. seisor": [1, 2],
    "5. bo": [3, 4],
    "2. lock and lol": [5, 6],
    "1. um": [7, 8],
    "4. mook": [9, 10],
}

ALL_GESTURES = ["3. seisor", "5. bo", "2. lock and lol", "1. um", "4. mook"]


# ---------------------------------------------------------------
# 난이도별 10라운드 출제 순서 생성
# ---------------------------------------------------------------
def _shuffle_no_adjacent_duplicates(pool, max_tries=200):
    """리스트를 섞되, 같은 값이 바로 옆에 연속되지 않도록 한다."""
    for _ in range(max_tries):
        random.shuffle(pool)
        if all(pool[i] != pool[i + 1] for i in range(len(pool) - 1)):
            return pool[:]
    return pool[:]  # 못 찾으면 마지막 셔플 결과라도 반환


def generate_round_sequence(difficulty):
    """
    난이도에 따라 10라운드의 (image_id, gesture_label) 순서를 만든다.
    difficulty: "easy" | "normal" | "hard"

    공통: 가위/보/락앤롤/엄지/바위 각 2장, 총 10장을 전부 한 번씩 사용한다.
    - easy   : 같은 모양이 2번씩 연속으로 나옴. 락앤롤 쌍은 맨 앞/뒤로 보내 비중을 낮춤.
    - normal : 같은 모양이 연속되지 않도록 번갈아 나옴. 락앤롤 등장 빈도는 그대로 1쌍.
    - hard   : 완전 무작위. 락앤롤-가위처럼 헷갈리는 조합이 연속으로 나올 수도 있음.
    """
    # 항상 5종 x 2장 = 10장을 모두 사용한다.
    pool = []
    for g in ALL_GESTURES:
        pool.extend([g, g])

    if difficulty == "easy":
        # "같은 모양 2연속" 쌍 단위를 만들고, 쌍의 등장 순서만 섞는다.
        # 락앤롤 쌍은 비중을 낮추는 취지로 맨 앞 또는 맨 뒤에 고정 배치.
        non_lockroll_pairs = [g for g in ALL_GESTURES if g != "2. lock and lol"]
        random.shuffle(non_lockroll_pairs)
        pair_order = non_lockroll_pairs + ["2. lock and lol"]  # 락앤롤은 마지막에
        if random.random() < 0.5:
            pair_order = ["2. lock and lol"] + non_lockroll_pairs  # 가끔은 맨 앞

        sequence = []
        for g in pair_order:
            sequence.extend([g, g])

    elif difficulty == "normal":
        # 같은 모양이 바로 옆에 연속되지 않도록 섞는다 (락앤롤 포함 5종 그대로 2장씩).
        sequence = _shuffle_no_adjacent_duplicates(pool)

    else:  # hard
        # 완전 무작위. 연속 중복이나 헷갈리는 조합(락앤롤-가위 등)이 그대로 나올 수 있음.
        sequence = pool[:]
        random.shuffle(sequence)

    sequence = sequence[:TOTAL_ROUNDS]

    # 각 제스처에 대해 이미지 ID(두 장 중 하나)를 골라 최종 라운드 목록 생성
    rounds = []
    for gesture in sequence:
        image_id = random.choice(IMAGE_IDS_BY_GESTURE[gesture])
        rounds.append((image_id, gesture))

    return rounds


# ---------------------------------------------------------------
# 손인식 래퍼 (hand_recognize.py 의 함수들을 그대로 사용)
# ---------------------------------------------------------------
class HandRecognizer:
    """
    카메라 1프레임을 읽고, hand_recognize.py 의 함수를 그대로 호출해
    (pip용 pygame.Surface, 인식된 제스처 라벨 또는 None) 을 반환한다.
    hand_recognize.py 의 cv2.imshow / 자체 while 루프는 절대 호출하지 않는다.
    """

    def __init__(self):
        ensure_model_downloaded()
        self.landmarker = create_landmarker()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결/권한(맥 보안 설정)을 확인해주세요.")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_idx = 0

    def process_frame(self):
        """
        한 프레임을 읽어 처리.
        반환값: (pip_surface, gesture_label)
          - pip_surface: PIP 표시용 pygame.Surface (None이면 프레임 읽기 실패)
          - gesture_label: classify_gesture()가 반환한 문자열, 손이 없으면 None
        """
        ok, frame = self.cap.read()
        if not ok:
            return None, None

        frame = cv2.flip(frame, 1)  # 셀카 모드처럼 좌우 반전 (hand_recognize.py와 동일)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((self.frame_idx / self.fps) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self.frame_idx += 1

        gesture_label = None
        if result.hand_landmarks:
            # 여러 손이 잡혀도 첫 번째 손만 판정에 사용
            landmarks = result.hand_landmarks[0]
            states = get_finger_states_by_angle(landmarks)
            gesture_label = classify_gesture(states)

        # OpenCV(BGR, numpy) 프레임을 pygame Surface로 변환
        # rgb_frame은 (H, W, 3) RGB 순서이므로 pygame이 바로 쓸 수 있게 축만 교환
        pip_surface = pygame.image.frombuffer(
            rgb_frame.tobytes(), (rgb_frame.shape[1], rgb_frame.shape[0]), "RGB"
        )
        return pip_surface, gesture_label

    def close(self):
        self.cap.release()


# ---------------------------------------------------------------
# 이미지 로드
# ---------------------------------------------------------------
def load_question_images():
    """1.jpg ~ 10.jpg 를 미리 로드해서 {번호: surface} 형태로 반환."""
    images = {}
    for i in range(1, 11):
        path = os.path.join(ASSETS_DIR, f"{i}.jpg")
        if not os.path.exists(path):
            print(f"[경고] 이미지가 없습니다: {path}")
            continue
        surface = pygame.image.load(path).convert()
        images[i] = surface
    return images


def fit_image_to_box(surface, box_w, box_h):
    """원본 비율을 유지하면서 box 안에 맞도록 축소/확대."""
    img_w, img_h = surface.get_size()
    scale = min(box_w / img_w, box_h / img_h)
    new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
    return pygame.transform.smoothscale(surface, new_size)


# ---------------------------------------------------------------
# 메인 게임 클래스
# ---------------------------------------------------------------
class Game:
    STATE_START = "START"
    STATE_PLAYING = "PLAYING"
    STATE_RESULT = "RESULT"

    def __init__(self):
        pygame.init()
        pygame.mixer.quit()  # ffpyplayer/OpenCV가 오디오를 따로 쓰지 않으므로 충돌 방지용으로 비활성화

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("손모양 따라하기 게임")
        self.clock = pygame.time.Clock()

        self.font_large = self._load_korean_font(60)
        self.font_medium = self._load_korean_font(36)
        self.font_small = self._load_korean_font(26)

        self.images = load_question_images()
        self.recognizer = HandRecognizer()

        self.state = self.STATE_START
        self.rounds = []
        self.current_round_idx = 0
        self.round_start_time = 0.0
        self.is_correct_now = False
        self.wrong_count = 0
        self.results_log = []  # 각 라운드 결과 (True/False) 기록

    @staticmethod
    def _load_korean_font(size):
        try:
            available = pygame.font.get_fonts()
            for name in ("applesdgothicneo", "AppleSDGothicNeo"):
                if name.lower().replace(" ", "") in available:
                    return pygame.font.SysFont(name, size)
        except Exception:
            pass
        return pygame.font.SysFont(None, size)

    # ----------------------- 대기 화면 -----------------------
    def draw_start_screen(self):
        self.screen.fill((25, 25, 30))

        title = self.font_large.render("손모양 따라하기 게임", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 160)))

        guide_lines = [
            "난이도를 선택하세요",
            "[1] 하   [2] 중   [3] 상",
            "(종료: 0)",
        ]
        for i, line in enumerate(guide_lines):
            surf = self.font_medium.render(line, True, (220, 220, 220))
            self.screen.blit(surf, surf.get_rect(center=(SCREEN_WIDTH // 2, 280 + i * 50)))

    # ----------------------- 라운드 시작 -----------------------
    def start_game(self, difficulty):
        self.rounds = generate_round_sequence(difficulty)
        self.current_round_idx = 0
        self.wrong_count = 0
        self.results_log = []
        self._start_round()
        self.state = self.STATE_PLAYING

    def _start_round(self):
        self.round_start_time = time.time()
        self.is_correct_now = False

    def _finish_current_round(self, correct):
        self.results_log.append(correct)
        if not correct:
            self.wrong_count += 1

        self.current_round_idx += 1
        if self.current_round_idx >= len(self.rounds):
            self.state = self.STATE_RESULT
        else:
            self._start_round()

    # ----------------------- 게임 화면 -----------------------
    def draw_playing_screen(self, pip_surface, gesture_label):
        self.screen.fill((25, 25, 30))

        image_id, target_gesture = self.rounds[self.current_round_idx]

        # 정답 이미지를 화면 중앙 큰 박스에 표시
        question_surface = self.images.get(image_id)
        if question_surface is not None:
            box_w, box_h = 480, 480
            fitted = fit_image_to_box(question_surface, box_w, box_h)
            rect = fitted.get_rect(center=(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 20))
            self.screen.blit(fitted, rect)

        # 현재 인식 결과가 정답과 일치하는지 판정
        self.is_correct_now = (gesture_label == target_gesture)

        # O / X 표시 (정답 이미지 옆)
        mark_color = (60, 200, 60) if self.is_correct_now else (220, 60, 60)
        mark_text = "O" if self.is_correct_now else "X"
        mark_surface = self.font_large.render(mark_text, True, mark_color)
        mark_rect = mark_surface.get_rect(center=(SCREEN_WIDTH // 2 + 260, SCREEN_HEIGHT // 2 - 20))
        self.screen.blit(mark_surface, mark_rect)

        # 라운드 진행 정보
        round_info = f"{self.current_round_idx + 1} / {len(self.rounds)}  라운드   틀림: {self.wrong_count}"
        info_surface = self.font_small.render(round_info, True, (200, 200, 200))
        self.screen.blit(info_surface, (30, 30))

        # 남은 시간 표시
        elapsed = time.time() - self.round_start_time
        remaining = max(0.0, ROUND_DURATION_SEC - elapsed)
        timer_surface = self.font_small.render(f"남은 시간: {remaining:0.1f}s", True, (200, 200, 200))
        self.screen.blit(timer_surface, (30, 65))

        # PIP(웹캠) 표시 - 우하단
        self._draw_pip(pip_surface)

        # 라운드 종료 체크 (정답이든 오답이든 시간이 지나면 다음으로)
        if elapsed >= ROUND_DURATION_SEC:
            self._finish_current_round(self.is_correct_now)

    def _draw_pip(self, pip_surface):
        # 카메라 영역 배경(테두리 느낌)
        border_rect = pygame.Rect(PIP_X - 4, PIP_Y - 4, PIP_WIDTH + 8, PIP_HEIGHT + 8)
        pygame.draw.rect(self.screen, (255, 255, 255), border_rect, border_radius=8)

        if pip_surface is not None:
            scaled = pygame.transform.scale(pip_surface, (PIP_WIDTH, PIP_HEIGHT))
            self.screen.blit(scaled, (PIP_X, PIP_Y))
        else:
            placeholder = pygame.Surface((PIP_WIDTH, PIP_HEIGHT))
            placeholder.fill((40, 40, 40))
            self.screen.blit(placeholder, (PIP_X, PIP_Y))

    # ----------------------- 결과 화면 -----------------------
    def draw_result_screen(self):
        self.screen.fill((25, 25, 30))

        correct_count = sum(1 for r in self.results_log if r)
        title = self.font_large.render("결과", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 140)))

        summary = f"정답 {correct_count} / {len(self.results_log)}   (틀림 {self.wrong_count}회)"
        summary_surface = self.font_medium.render(summary, True, (220, 220, 220))
        self.screen.blit(summary_surface, summary_surface.get_rect(center=(SCREEN_WIDTH // 2, 220)))

        # 라운드별 O/X 한눈에 보기
        start_x = SCREEN_WIDTH // 2 - (len(self.results_log) * 50) // 2
        for i, correct in enumerate(self.results_log):
            color = (60, 200, 60) if correct else (220, 60, 60)
            mark = "O" if correct else "X"
            mark_surface = self.font_medium.render(mark, True, color)
            self.screen.blit(mark_surface, (start_x + i * 50, 300))

        guide = self.font_small.render("다시 시작: 1(하)/2(중)/3(상)   종료: 0", True, (200, 200, 200))
        self.screen.blit(guide, guide.get_rect(center=(SCREEN_WIDTH // 2, 420)))

    # ----------------------- 이벤트 처리 -----------------------
    def handle_keydown(self, key):
        if key == pygame.K_0:
            return False  # 종료 신호

        if self.state in (self.STATE_START, self.STATE_RESULT):
            if key == pygame.K_1:
                self.start_game("easy")
            elif key == pygame.K_2:
                self.start_game("normal")
            elif key == pygame.K_3:
                self.start_game("hard")

        return True

    # ----------------------- 메인 루프 -----------------------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_keydown(event.key)

            # 카메라는 항상 처리 (PIP는 START/RESULT 화면에서도 보여줄지 결정 가능,
            # 여기서는 PLAYING 상태일 때만 인식 결과를 게임 판정에 사용)
            pip_surface, gesture_label = self.recognizer.process_frame()

            if self.state == self.STATE_START:
                self.draw_start_screen()
                self._draw_pip(pip_surface)
            elif self.state == self.STATE_PLAYING:
                self.draw_playing_screen(pip_surface, gesture_label)
            elif self.state == self.STATE_RESULT:
                self.draw_result_screen()
                self._draw_pip(pip_surface)

            pygame.display.flip()
            self.clock.tick(30)

        self.recognizer.close()
        pygame.quit()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
실행 진입점.
실제 게임 로직(화면/난이도/인식 판정)은 launch_screen.py 에 있습니다.
손 인식 자체의 핵심 로직은 hand_recognize.py 에 있으며, 그 파일은 수정하지 않고
launch_screen.py 에서 함수만 가져와 사용합니다.
"""

from launch_screen import main

if __name__ == "__main__":
    main()

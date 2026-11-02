# AR 시각화·UI와 예시 실행

## 이번 변경

예시 맵의 출발점은 초록 삼각형, 도착점은 빨간 과녁으로 표시합니다. 사용자 제공 이미지의 아이콘 부분만 사용해 마커 아래 글씨와 사각 테두리를 제외했습니다. 장애물은 칸 전체를 검게 채우며 격자선은 유지합니다. 파란 AR 캐릭터만 움직이고 출발점·도착점·장애물은 고정됩니다.

기본 예제는 수직 촬영 형태의 5×5 격자입니다. 가로·세로 칸 수가 달라도 각 칸은 정사각형이며 카드 상자는 맵 아래에 35:45 비율로 밀착 배치합니다. 실제 촬영 영상의 마커를 바꾸는 기능은 아닙니다.

## 설치와 실행

Python 3.10 이상에서 저장소 폴더를 기준으로 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m examples.demo_ar
```

Windows에서는 준비 후 `run_ar_demo.cmd`를 더블클릭해도 됩니다.

- Space 또는 시작 버튼: 재생
- 1: 성공 / 2: 장애물 충돌 / 3: 맵 이탈
- 4: 반복문 문법 오류 / 5: 반복 실행 / 6: 좌우 회전
- Esc: 종료

장애물이 있는 **2번 예제**에서 세 표식을 함께 볼 수 있습니다.

```powershell
# 가변 맵 크기: rows는 세로, cols는 가로
.\.venv\Scripts\python.exe -m examples.demo_ar --rows 5 --cols 7
.\.venv\Scripts\python.exe -m examples.demo_ar --rows 9 --cols 9

# 추가 원근 변환 검사
.\.venv\Scripts\python.exe -m examples.demo_ar --layout perspective

# 자동 검사
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

이 예제의 맵과 이동 기록은 테스트 데이터입니다. 이미지 에셋은 화면 표시용이고, 영문 약어 카드 역시 AR 강조 검사용입니다. 실제 인쇄·인식에는 팀에서 정한 원본 인쇄물을 사용합니다.

## 촬영 사진을 배경으로 AR만 검사하기

```powershell
.\.venv\Scripts\python.exe -m examples.demo_ar --image "C:\사진\map.jpg" --rows 5 --cols 5
```

이 모드는 사진의 격자 외곽 네 꼭짓점을 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래 순서로 클릭하고 Enter로 확정합니다. 지정한 가로·세로 칸 수와 테스트용 이동 기록을 사용합니다. 실제 시작점/도착점/장애물/카드 인식 검사가 아닙니다.

`--auto-grid`와 `examples/grid_size.py`는 별도의 기하 실험용 칸 수 추정 기능입니다. 팀의 `modules/map_recognition.py`를 대체하지 않으며 기본 예시 실행에 사용하지 않습니다.

개인 PC의 임시 폴더에 있는 팀 맵 코드 복사본을 이용한 사진 실험 도구와 결과 파일은 이번 커밋에 포함하지 않습니다.

## 공통 인터페이스와 최종 통합

공통 모델과 AR 공개 메서드의 인자·반환 형식을 유지했습니다. AR의 기존 표시 로직은 그대로이며 규격에 관한 주석을 보완했습니다.

```python
from modules.ar_visualization import ARVisualizer
ar = ARVisualizer()
ar.begin(map_data, cards, result)  # 실행 시작 때 한 번
output_frame = ar.render(latest_frame)  # 카메라에서 읽은 새 프레임마다 호출
finished = ar.is_finished()
clicked = ar.is_start_clicked(x, y, latest_frame.shape)
```

2026-11-02 확인한 노션 기획서와 인쇄물 규격에 따른 연결 조건입니다.

- A4 가로 용지, 동일한 정사각형 칸, 가변 맵 크기, 초기 방향 위쪽을 사용합니다.
- 카메라와 종이는 수직 촬영 위치에 고정합니다. 맵은 위쪽, 카드는 아래쪽에 겹치지 않게 놓고 전체가 한 화면에 들어오게 합니다.
- Start 시점의 **원본 프레임**을 맵 및 카드 인식에 전달하고 명령 처리 모듈의 실행 결과를 받습니다. AR/UI가 그려진 출력 영상은 인식 입력으로 재사용하지 않습니다.
- 초기 CharacterState의 위치와 Direction.UP은 명령 처리 모듈이 생성합니다. AR은 before/after/final_state를 그대로 재생합니다.
- begin은 실행 시작 때 한 번, render는 최신 카메라 프레임마다 호출합니다. 카드 수정 후 Start를 누르면 재인식하고 새 실행 결과로 begin을 호출합니다.
- corners는 종이 모서리가 아닌 격자 외곽입니다. frame, corners, CardData.bbox, 클릭 좌표는 같은 해상도·좌표계를 사용해야 합니다.
- begin에서 좌표를 복사하므로 카메라/종이를 움직이거나 프레임 크기를 변경했다면 다시 인식한 뒤 시작해야 합니다. 실시간 종이 추적 기능은 없습니다.
- 맵 검출·카드 검출·명령 해석·성공 및 충돌 판정·카메라 수집은 각 담당 모듈에서 수행합니다.
- render는 원본과 같은 크기·자료형의 새 uint8 BGR 이미지를 반환하고 원본을 수정하지 않습니다.
- 한글 표시에는 Pillow>=10.0과 한글 글꼴이 필요합니다. Windows는 맑은 고딕을 사용합니다. 다른 한글 글꼴 경로는 CARDROAD_FONT_PATH로 지정합니다.

노션에 적힌 fillPoly/addWeighted/putText는 예상 함수입니다. 구현에서는 삼각형을 fillConvexPoly로 그리고 한글은 Pillow로 표시합니다. 전진 0.6초·회전 0.4초는 현재 UX 설정이며 기획서 지정값은 아닙니다.

## 검증 및 한계

자동 검사 33개를 통과했습니다. 가변 크기(5×5, 5×7, 7×5, 9×9 등), 초기 방향·좌표 변환, 이동/회전 보간, 밀착 카드 및 반복 강조, 오류 표시, 재실행, 원본 프레임 보존, 한글 표시를 검사합니다. 새 프레임 반영과 process_game 재실행 검사는 합성 프레임 및 인식/엔진 대역을 사용합니다.

별도 로컬 환경에서는 공유 사진 picture_5~8의 맵 인식 결과와 테스트용 이동 기록을 연결했습니다. picture_1~3은 필수 마커 미검출, picture_4는 GRID_NOT_FOUND로 재생에서 제외했습니다. 이는 정답 라벨과 비교한 인식 정확도가 아닙니다.

실제 원본 인쇄물, 카드 인식·명령 처리 모듈, main.py 웹캠 루프를 함께 연결한 전체 시스템 검증은 남아 있습니다. 이 변경에 다른 팀원 모듈이나 main.py의 구현 변경은 포함하지 않습니다.
from unittest import result

import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ===== 모델 경로 설정 =====
model_path = "C:/Users/User/Downloads/hand_landmarker.task"

# ===== Mediapipe 기본 옵션 =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# ===== 시각화 함수 (랜드마크 그리기) =====
def draw_landmarks_on_image(rgb_image, detection_result):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    annotated_image = rgb_image.copy()

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Mediapipe용 proto 포맷으로 변환
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return annotated_image

# ===== 결과 콜백 함수 (비동기 결과 수신 시 호출됨) =====
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result  # 최신 결과를 전역 변수로 저장
    latest_result = result

# ===== HandLandmarker 옵션 설정 =====
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2
)

# ===== 웹캠 실행 =====
cap = cv2.VideoCapture(0)
latest_result = None  # 최신 탐지 결과 저장용 변수

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # (1) Mediapipe용 RGB 이미지로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # (2) 프레임 타임스탬프 (밀리초)
        frame_timestamp_ms = int(time.time() * 1000)

        # (3) mp.Image 객체 생성
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # (4) 비동기 추론
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # (5) 최신 결과가 있으면 시각화
        if latest_result is not None:
            annotated_frame = draw_landmarks_on_image(rgb_frame, latest_result)
            # Mediapipe는 RGB → OpenCV는 BGR 이므로 다시 변환
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Hand Tracking", annotated_frame)
        else:
            cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
            break

cap.release()
cv2.destroyAllWindows()

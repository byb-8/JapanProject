import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 랜드마크 그리기
            for lm in face_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Head pose 계산 (PnP)
            # 대표 포인트: 코(1), 턱(152), 왼쪽 눈(33), 오른쪽 눈(263), 입 왼쪽(61), 입 오른쪽(291)
            image_points = np.array([
                (face_landmarks.landmark[1].x * frame.shape[1],
                 face_landmarks.landmark[1].y * frame.shape[0]),    # 코 끝
                (face_landmarks.landmark[152].x * frame.shape[1],
                 face_landmarks.landmark[152].y * frame.shape[0]),  # 턱
                (face_landmarks.landmark[33].x * frame.shape[1],
                 face_landmarks.landmark[33].y * frame.shape[0]),   # 왼쪽 눈
                (face_landmarks.landmark[263].x * frame.shape[1],
                 face_landmarks.landmark[263].y * frame.shape[0]),  # 오른쪽 눈
                (face_landmarks.landmark[61].x * frame.shape[1],
                 face_landmarks.landmark[61].y * frame.shape[0]),   # 왼쪽 입
                (face_landmarks.landmark[291].x * frame.shape[1],
                 face_landmarks.landmark[291].y * frame.shape[0])   # 오른쪽 입
            ], dtype=np.float64)

            model_points = np.array([
                (0.0, 0.0, 0.0),          # 코 끝
                (0.0, -63.6, -12.5),      # 턱
                (-43.3, 32.7, -26.0),     # 왼쪽 눈
                (43.3, 32.7, -26.0),      # 오른쪽 눈
                (-28.9, -28.9, -24.1),    # 왼쪽 입
                (28.9, -28.9, -24.1)      # 오른쪽 입
            ])

            focal_length = frame.shape[1]
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                      [0, focal_length, center[1]],
                                      [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            # Euler angles
            rmat, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
            pitch = np.arctan2(-rmat[2,0], sy)
            yaw = np.arctan2(rmat[1,0], rmat[0,0])
            roll = np.arctan2(rmat[2,1], rmat[2,2])

            cv2.putText(frame, f"Yaw: {np.degrees(yaw):.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(frame, f"Pitch: {np.degrees(pitch):.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.putText(frame, f"Roll: {np.degrees(roll):.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

    cv2.imshow("Face Mesh + Head Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

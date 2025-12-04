import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe.framework.formats import landmark_pb2
img=cv2.imread("C:/Users/User/Desktop/20151231000271_0.jpg")
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/User/Downloads/hand_landmarker.task"),
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE)
with HandLandmarker.create_from_options(options) as landmarker:
    numpy_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    hand_landmarker_result = landmarker.detect(mp_image)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

for hand_landmarks in hand_landmarker_result.hand_landmarks:

    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
    ])

    mp_drawing.draw_landmarks(
        image=numpy_image,
        landmark_list=hand_landmarks_proto,
        connections=mp.solutions.hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
    )
numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
cv2.imshow("Hand Landmarks", numpy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
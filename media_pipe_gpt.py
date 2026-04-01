import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the downloaded model
model_path = "pose_landmarker_full.task"

# Setup options
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

# Create landmarker
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# External camera (1 or 2 usually)
cap = cv2.VideoCapture(0)

frame_timestamp = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect pose
    result = pose_landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if result.pose_landmarks:

        for landmark in result.pose_landmarks[0]:
            h, w, _ = frame.shape
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 5, (0,255,0), -1)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
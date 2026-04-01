import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Pose model path
model_path = "pose_landmarker_lite.task"

# Create landmarker
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# External camera index
cap = cv2.VideoCapture(1)
print(cap.isOpened())

frame_timestamp = 0

# Pose skeleton connections (MediaPipe format)
POSE_CONNECTIONS = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (23,24),
    (11,23),(12,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = pose_landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]
        h, w, _ = frame.shape

        points = []

        # Draw landmarks
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 4, (0,255,0), -1)

        # Draw skeleton connections
        for connection in POSE_CONNECTIONS:
            start = connection[0]
            end = connection[1]

            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0,255,255), 2)

    cv2.imshow("MediaPipe Pose Skeleton", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
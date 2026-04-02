import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ======================
# Load models
# ======================
pose_model = "pose_landmarker_lite.task"
hand_model = "hand_landmarker.task"

base_pose = python.BaseOptions(model_asset_path=pose_model)
base_hand = python.BaseOptions(model_asset_path=hand_model)

pose_options = vision.PoseLandmarkerOptions(
    base_options=base_pose,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

hand_options = vision.HandLandmarkerOptions(
    base_options=base_hand,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

pose = vision.PoseLandmarker.create_from_options(pose_options)
hands = vision.HandLandmarker.create_from_options(hand_options)

# ======================
# Camera
# ======================
cap = cv2.VideoCapture(2)
timestamp = 0

# Pose connections
POSE_CONNECTIONS = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (23,24),
    (11,23),(12,24)
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Draw divider line
    cv2.line(frame, (w//2, 0), (w//2, h), (255,255,255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ======================
    # Run models
    # ======================
    pose_result = pose.detect_for_video(mp_image, timestamp)
    hand_result = hands.detect_for_video(mp_image, timestamp)
    timestamp += 1

    # ======================
    # LEFT SIDE → HANDS
    # ======================
    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:

            # Get wrist (landmark 0)
            wrist = hand_landmarks[0]
            x_px = int(wrist.x * w)

            # Only process if on LEFT side
            if x_px < w // 2:
                points = []

                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Draw simple hand connections
                HAND_CONNECTIONS = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20)
                ]

                for c in HAND_CONNECTIONS:
                    cv2.line(frame, points[c[0]], points[c[1]], (0,255,255), 2)

                cv2.putText(frame, "HAND ZONE", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # ======================
    # RIGHT SIDE → POSE
    # ======================
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        points = []

        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        # Check torso center (mid shoulders)
        left_shoulder = points[11]
        right_shoulder = points[12]
        torso_x = (left_shoulder[0] + right_shoulder[0]) // 2

        if torso_x >= w // 2:

            # Draw landmarks
            for p in points:
                cv2.circle(frame, p, 4, (255,0,0), -1)

            # Draw pose skeleton
            for c in POSE_CONNECTIONS:
                cv2.line(frame, points[c[0]], points[c[1]], (255,255,0), 2)

            cv2.putText(frame, "POSE ZONE", (w//2 + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # ======================
    # Display
    # ======================
    cv2.imshow("Gesture Split System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
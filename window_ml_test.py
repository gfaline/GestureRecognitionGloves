import numpy as np
import joblib
from collections import deque

WINDOW_SIZE = 20

model = joblib.load("gesture_model_windowed.pkl")
scaler = joblib.load("scaler_windowed.pkl")

buffer = deque(maxlen=WINDOW_SIZE)


def predict_realtime(new_sample):
    buffer.append(new_sample)

    if len(buffer) < WINDOW_SIZE:
        return None  # not enough data yet

    window = np.array(buffer).flatten().reshape(1, -1)
    window = scaler.transform(window)

    return model.predict(window)[0]
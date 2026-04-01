import pandas as pd
import numpy as np
import joblib
import time
import random
from collections import deque, defaultdict

# =========================
# CONFIG
# =========================
WINDOW_SIZE = 25
STRIDE = 10

# =========================
# STREAMER
# =========================
class RealtimeSensorStreamer:
    def __init__(self, csv_path, time_scale=1.0, noise_ratio=(5, 20)):
        self.df = pd.read_csv(csv_path, comment='#')
        self.time_scale = time_scale
        self.noise_ratio = noise_ratio

        self.gesture_chunks = []
        self.noise_pool = []

        self._prepare_chunks()

    def _prepare_chunks(self):
        current_chunk = []

        for _, row in self.df.iterrows():
            if row['Button'] == 1:
                current_chunk.append(row)
            else:
                if current_chunk:
                    self.gesture_chunks.append(pd.DataFrame(current_chunk))
                    current_chunk = []
                self.noise_pool.append(row)

        if current_chunk:
            self.gesture_chunks.append(pd.DataFrame(current_chunk))

    def _get_noise_block(self):
        n = random.randint(*self.noise_ratio)
        return random.choices(self.noise_pool, k=n)

    def stream(self):
        while True:
            random.shuffle(self.gesture_chunks)

            for gesture in self.gesture_chunks:
                # noise first
                for row in self._get_noise_block():
                    yield row.to_dict()
                    time.sleep(0.05 / self.time_scale)

                # then gesture (preserve timing)
                prev_t = None
                for _, row in gesture.iterrows():
                    data = row.to_dict()

                    if prev_t is not None:
                        dt = (data['time_ms'] - prev_t) / 1000.0
                        if dt <= 0 or dt > 1:
                            dt = 0.05
                    else:
                        dt = 0.05

                    yield data
                    time.sleep(dt / self.time_scale)

                    prev_t = data['time_ms']


# =========================
# WINDOW BUFFER
# =========================

class SlidingWindow:
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

        self.feature_buffer = deque(maxlen=size)
        self.label_buffer = deque(maxlen=size)

        self.counter = 0

        # Explicit feature list (IMPORTANT)
        self.feature_keys = [
            'norm0', 'norm1', 'norm2', 'norm3', 'norm4',
            'gyroX_dps', 'gyroY_dps', 'gyroZ_dps'
        ]

    def add(self, sample):
        # -------------------------
        # Extract features ONLY
        # -------------------------
        features = [sample[k] for k in self.feature_keys]

        # Store features + label separately
        self.feature_buffer.append(features)
        self.label_buffer.append(sample['gestureID'])

        self.counter += 1

        # -------------------------
        # Window not full yet
        # -------------------------
        if len(self.feature_buffer) < self.size:
            return None

        # -------------------------
        # Stride check
        # -------------------------
        if self.counter % self.stride != 0:
            return None

        # -------------------------
        # Label consistency check
        # -------------------------
        labels = list(self.label_buffer)
        if len(set(labels)) != 1:
            return None

        # -------------------------
        # Build clean window
        # -------------------------
        window_features = list(self.feature_buffer)
        label = labels[0]

        return window_features, label


# =========================
# MODEL TEST HARNESS
# =========================
class ModelTester:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler

        self.correct = 0
        self.total = 0

        # Per-label stats
        self.label_total = defaultdict(int)
        self.label_correct = defaultdict(int)

    def process_window(self, window, label):
        # Build feature matrix
        """features = np.array([
            [
                w['raw0'], w['raw1'], w['raw2'],
                w['raw3'], w['raw4'],
                w['gyroX_dps'], w['gyroY_dps'], w['gyroZ_dps']
            ]
            for w in window
        ])"""

        features = np.array(window)

        #label = label[0]

        # Flatten (VERY likely how your RF was trained)
        x = features.flatten().reshape(1, -1)

        # Apply scaler if exists
        if self.scaler is not None:
            x = self.scaler.transform(x)

        # Predict
        pred = self.model.predict(x)[0]


        # Metrics
        self.total += 1
        self.label_total[label] += 1
        if pred == label:
            self.correct += 1
            self.label_correct[label] += 1

        acc = self.correct / self.total

        print(f"GT: {label} | Pred: {pred} | Acc: {acc:.3f}")


# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    streamer = RealtimeSensorStreamer("CSV log 1.txt")
    window = SlidingWindow(WINDOW_SIZE, STRIDE)

    scaler_filename = 'win_gesture_scaler1.pkl'
    model_filename = 'win_regression_gesture_model1.pkl'
    try:
        scaler = joblib.load(scaler_filename)
        print(f"Scaler '{scaler_filename}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {scaler_filename} not found.")
        exit()

    try:
        model = joblib.load(model_filename)
        print(f"Model '{model_filename}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {model_filename} not found.")
        exit()

    """# First, scale the new data using the loaded scaler's transform method
scaled_new_data = loaded_scaler.transform(new_data)

# Then, make predictions using the loaded model
predictions = loaded_model.predict(scaled_new_data)"""
    tester = ModelTester(model, scaler)

    try:
        for sample in streamer.stream():
            w = window.add(sample)

            if w is not None:
                win, label = w
                tester.process_window(win, label)
    except KeyboardInterrupt:
        print("\n\n=== Evaluation Summary ===")

        # Overall accuracy
        if tester.total > 0:
            overall_acc = tester.correct / tester.total
            print(f"Overall Accuracy: {overall_acc:.3f}")
            print(f"Total Samples: {tester.total}")
        else:
            print("No samples processed.")

        print("\nPer-label performance:")

        for label in sorted(tester.label_total.keys()):
            total = tester.label_total[label]
            correct = tester.label_correct[label]

            acc = correct / total if total > 0 else 0.0

            print(f"Label {label}: {correct}/{total} ({acc:.3f})")
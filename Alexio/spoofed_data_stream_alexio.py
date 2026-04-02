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
THRESHOLD = 0.7
IDLE_CLASS = 0

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

        # For having smaller continuous streams
        self.last_label = None
        self.same_label_windows = 0

        self.spike_prob = 0.02  # 2% chance per sample
        self.drift_prob = 0.01  # occasional drift
        self.false_gesture_prob = 0.005

        self.drift_offset = {k: 0 for k in [
            'norm0', 'norm1', 'norm2', 'norm3', 'norm4',
            'gyroX_dps', 'gyroY_dps', 'gyroZ_dps'
        ]}

    def _inject_noise(self, data):
        data = data.copy()

        sensor_keys = [
            'norm0', 'norm1', 'norm2', 'norm3', 'norm4',
            'gyroX_dps', 'gyroY_dps', 'gyroZ_dps'
        ]

        # -------------------------
        # 1. Spike noise (sharp glitch)
        # -------------------------
        if random.random() < self.spike_prob:
            key = random.choice(sensor_keys)
            data[key] += random.uniform(-50, 50)  # large spike

        # -------------------------
        # 2. Drift (slow bias)
        # -------------------------
        if random.random() < self.drift_prob:
            for k in sensor_keys:
                self.drift_offset[k] += random.uniform(-0.5, 0.5)

        for k in sensor_keys:
            data[k] += self.drift_offset[k]

        # -------------------------
        # 3. False gesture pattern
        # -------------------------
        """if random.random() < self.false_gesture_prob:
            # simulate coordinated motion (very dangerous for ML)
            for k in ['gyroX_dps', 'gyroY_dps', 'gyroZ_dps']:
                data[k] += random.uniform(10, 30)

            for k in ['raw0', 'raw1', 'raw2']:
                data[k] += random.uniform(5, 15)"""

        return data

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

    def notify_window(self, label):

        if label == self.last_label:
            self.same_label_windows += 1
        else:
            self.same_label_windows = 1
            self.last_label = label

    def _should_interrupt(self):
        max_windows = random.randint(3, 5)
        return self.same_label_windows >= max_windows

    def _get_noise_block(self):
        n = random.randint(*self.noise_ratio)
        return random.choices(self.noise_pool, k=n)

    def stream(self):
        while True:
            random.shuffle(self.gesture_chunks)

            for gesture in self.gesture_chunks:
                # noise first
                if self._should_interrupt():
                    for row in self._get_noise_block():
                        yield row.to_dict()
                        time.sleep(0.05)
                    self.same_label_windows = 0
                    self.last_label = None

                # then gesture (preserve timing)
                prev_t = None
                for _, row in gesture.iterrows():
                    if self._should_interrupt():
                        break
                    data = row.to_dict()

                    if prev_t is not None:
                        dt = (data['time_ms'] - prev_t) / 1000.0
                        if dt <= 0 or dt > 1:
                            dt = 0.05
                    else:
                        dt = 0.05

                    data = self._inject_noise(data)
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
        self.button_buffer = deque(maxlen=size)

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
        self.button_buffer.append(sample['Button'])

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
        buttons = list(self.button_buffer)
        if len(set(labels)) != 1:
            return None

        # -------------------------
        # Build clean window
        # -------------------------
        window_features = list(self.feature_buffer)

        return window_features, labels[0], buttons[0]


# =========================
# MODEL TEST HARNESS
# =========================
class ModelTester:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler

        self.correct = 0
        self.total = 0
        self.label_total = defaultdict(int)
        self.label_correct = defaultdict(int)

        # Per-label stats
        self.label_total = defaultdict(int)
        self.label_correct = defaultdict(int)

        #False pos stats
        self.noise_windows = 0
        self.false_positives = 0

        self.null_response = 0

    def process_window(self, window, label, button):
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

        try:
            pred_prob = self.model.predict_proba(x)[0]
            confidence = pred_prob.max()
        except Exception as e:
            print (e)
            confidence = 1

        if confidence < THRESHOLD:
            pred = IDLE_CLASS

        # -------------------------
        # 🔥 False Positive Logic
        # -------------------------
        if button == 0:
            label = IDLE_CLASS
            self.noise_windows += 1
            if confidence >= THRESHOLD and pred != IDLE_CLASS:
                self.false_positives += 1

        if button == 1 and confidence < THRESHOLD:
            self.null_response += 1

        # Metrics
        self.total += 1
        self.label_total[label] += 1
        if pred == label:
            self.correct += 1
            self.label_correct[label] += 1

        acc = self.correct / self.total

        print(f"GT: {label} | Pred: {pred} | Conf: {confidence} | Acc: {acc:.3f}")


# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    streamer = RealtimeSensorStreamer("CSV log 1.txt")
    window = SlidingWindow(WINDOW_SIZE, STRIDE)

    scaler_filename = 'alex_win_gesture_scaler1.pkl'
    model_filename = 'alex_win_regression_gesture_model1.pkl'
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
            result = window.add(sample)

            if result:
                features, label, button = result

                # 🔥 IMPORTANT: inform streamer
                streamer.notify_window(label)

                tester.process_window(features, label, button)
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
        print("\nFalse Positive Metrics:")
        if tester.noise_windows > 0:
            fpr = tester.false_positives / tester.noise_windows
            print(f"Noise windows: {tester.noise_windows}")
            print(f"False positives: {tester.false_positives}")
            print(f"False positive rate: {fpr:.3f}")
        else:
            print("No noise windows observed.")
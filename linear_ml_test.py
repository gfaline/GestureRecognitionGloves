import numpy as np
import joblib

model = joblib.load("gesture_model1.pkl")
scaler = joblib.load("gesture_scaler1.pkl")

def predict(sample):
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    return model.predict(sample)[0]

# Example usage
# sample = [raw0, raw1, raw2, raw3, raw4, gyroX, gyroY, gyroZ, norm0, ...]
gesture = predict(sample)
print("Gesture:", gesture)
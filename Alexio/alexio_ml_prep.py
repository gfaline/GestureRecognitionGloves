from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import numpy as np
import pandas as pd

df = pd.read_csv("filtered_button_not_zero.csv", comment='#')

# Separate features and labels
#X = df.drop(columns=["gestureID", "Button", "time_ms"]).values
X = df.drop(columns=["gestureID", "Button", "time_ms", "raw0", "raw1", "raw2", "raw3", "raw4"]).values
y = df["gestureID"].values
print(df["gestureID"].value_counts())

WINDOW_SIZE = 25
STEP_SIZE = 10


def create_windows_strict(X, y, window_size, step):
    Xw, yw = [], []

    for i in range(0, len(X) - window_size, step):
        window_y = y[i:i + window_size]

        # Check if all labels in window are the same
        if np.all(window_y == window_y[0]):
            window_X = X[i:i + window_size]

            Xw.append(window_X.flatten())
            yw.append(window_y[0])  # safe since all same

    return np.array(Xw), np.array(yw)


def create_windows_majority(X, y, window_size, step, threshold=0.8):
    Xw, yw = [], []

    for i in range(0, len(X) - window_size, step):
        window_y = y[i:i + window_size]

        values, counts = np.unique(window_y, return_counts=True)
        majority_label = values[np.argmax(counts)]
        ratio = np.max(counts) / window_size

        if ratio >= threshold:
            window_X = X[i:i + window_size]
            Xw.append(window_X.flatten())
            yw.append(majority_label)

    return np.array(Xw), np.array(yw)


#Xw, yw = create_windows_strict(X, y, WINDOW_SIZE, STEP_SIZE)
Xw, yw = create_windows_majority(X, y, WINDOW_SIZE, STEP_SIZE)

print("Windowed shape:", Xw.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

winX_train, winX_test, winy_train, winy_test = train_test_split(
    Xw, yw, test_size=0.2, random_state=42, stratify=yw
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

winscaler = StandardScaler()
winX_train = winscaler.fit_transform(winX_train)
winX_test = winscaler.transform(winX_test)

from sklearn.linear_model import LogisticRegression

line_reg_model = LogisticRegression(max_iter=1000)
line_reg_model.fit(X_train, y_train)

win_reg_model = LogisticRegression(max_iter=1000)
win_reg_model.fit(winX_train, winy_train)

win_forest_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

win_forest_model.fit(winX_train, winy_train)

#Testing reports
y_pred = line_reg_model.predict(X_test)
print(classification_report(y_test, y_pred))

win_y_pred = win_reg_model.predict(winX_test)
print(classification_report(winy_test, win_y_pred))

win_y_pred = win_forest_model.predict(winX_test)
print(classification_report(winy_test, win_y_pred))

joblib.dump(scaler, "alex_line_gesture_scaler1.pkl")
joblib.dump(winscaler, "alex_win_gesture_scaler1.pkl")

joblib.dump(line_reg_model, "alex_line_regression_gesture_model1.pkl")

joblib.dump(win_reg_model, "alex_win_regression_gesture_model1.pkl")
joblib.dump(win_reg_model, "alex_win_forest_gesture_model1.pkl")
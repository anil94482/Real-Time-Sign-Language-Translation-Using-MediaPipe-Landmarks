# train_model.py (updated)
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_DIR = "data"

def safe_load_npy(path):
    try:
        arr = np.load(path, allow_pickle=False)
        # ensure numeric
        arr = np.asarray(arr, dtype=np.float32)
        return arr
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return None

# Step 1: read all files (raw)
raw_entries = []   # list of (label, path, array or None)
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            arr = safe_load_npy(path)
            raw_entries.append((label, path, arr))

if not raw_entries:
    raise SystemExit("No .npy files found under data/ — check DATA_DIR and folder structure.")

# Step 2: inspect shapes (use flattened length)
lengths = []
for label, path, arr in raw_entries:
    if arr is None:
        lengths.append(None)
    else:
        lengths.append(arr.ravel().shape[0])

# Find the most common non-None length to be our target
valid_lengths = [L for L in lengths if L is not None]
if not valid_lengths:
    raise SystemExit("No valid numpy arrays were loaded (all failed).")

most_common_len = Counter(valid_lengths).most_common(1)[0][0]
print(f"ℹ️ Most common flattened length: {most_common_len}")

# Step 3: build X, y by padding/truncating flattened arrays to most_common_len
X_list, y_list = [], []
skipped = 0
padded = 0
truncated = 0

for (label, path, arr), L in zip(raw_entries, lengths):
    if arr is None:
        skipped += 1
        continue
    flat = arr.ravel().astype(np.float32)
    if flat.shape[0] == most_common_len:
        X_list.append(flat)
        y_list.append(label_map[label])
    elif flat.shape[0] < most_common_len:
        # pad with zeros
        pad = np.zeros(most_common_len - flat.shape[0], dtype=np.float32)
        X_list.append(np.concatenate([flat, pad]))
        y_list.append(label_map[label])
        padded += 1
    else:
        # truncate
        X_list.append(flat[:most_common_len])
        y_list.append(label_map[label])
        truncated += 1

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

print(f"✅ Loaded {len(X)} samples from {len(labels)} classes: {labels}")
print(f"Summary: skipped={skipped}, padded={padded}, truncated={truncated}")

# Quick safety checks
if len(np.unique(y)) < 2:
    raise SystemExit("Need at least 2 classes to train a classifier.")

# Step 4: split (use stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: train
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate
acc = clf.score(X_test, y_test)
print(f"🎯 Model accuracy: {acc*100:.2f}%")

# Save model + labels (save label_map for clarity)
joblib.dump(clf, "sign_model.pkl")
np.save("labels.npy", labels)        # list of class names in order
joblib.dump(label_map, "label_map.pkl")

print("✅ Model saved as sign_model.pkl")
print("✅ Labels saved as labels.npy")
print("✅ Label map saved as label_map.pkl")

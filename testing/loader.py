import os
from sklearn.feature_extraction.text import CountVectorizer
import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_dataset(benign_dir, malignant_dir, img_size=(128, 128), vectorize=True):
    """
    Loads benign and malignant image datasets from their directories.
    
    Args:
        benign_dir (str): Path to 'Benign' folder.
        malignant_dir (str): Path to 'Malignant' folder.
        img_size (tuple): Size to which each image will be resized.
        vectorize (bool): If True, flatten images into feature vectors.
    
    Returns:
        X (np.ndarray): Image data (n_samples, features)
        y (np.ndarray): Labels (0 for benign, 1 for malignant)
        vectorizer (StandardScaler): Fitted scaler for normalization
    """
    
    X = []
    y = []
    
    # --- Load benign images ---
    for filename in os.listdir(benign_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(benign_dir, filename)
            img = cv2.imread(img_path)
            if img is None: 
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(0)  # Benign = 0
    
    # --- Load malignant images ---
    for filename in os.listdir(malignant_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(malignant_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(1)  # Malignant = 1
    
    # Convert to NumPy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # --- Optional: flatten and normalize ---
    if vectorize:
        X = X.reshape(X.shape[0], -1)
        vectorizer = StandardScaler()
        X = vectorizer.fit_transform(X)
    else:
        vectorizer = None
    
    print(f"✅ Loaded {len(X)} images ({len(y[y==0])} benign, {len(y[y==1])} malignant)")
    return X, y, vectorizer


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2, imagenet_utils
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from randomforest_base_functions import qigpso_feature_selection
from imageclasssifcation import *

def run_qigpso_on_images(df, popsize=20, alpha=0.8, max_iter=100, g0=35, test_size=0.2, random_seed=42, verbose=False):
    """
    df: dataframe with 'image_path' and 'label'
    Returns: selected features mask, selected features array, fitness, test accuracy
    """
    X_features, y = extractimagefeatures(df)
    # standardize features
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    best_mask, best_fitness = qigpso_feature_selection(
        X_train, y_train, popsize, alpha, max_iter, g0, random_seed=random_seed, verbose=verbose
    )
    
    X_train_selected = X_train[:, best_mask==1]
    X_test_selected = X_test[:, best_mask==1]

    clf = RandomForestClassifier(n_estimators=200, random_state=random_seed)
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    test_acc = accuracy_score(y_test, y_pred)
    
    return best_mask, X_train_selected, best_fitness, test_acc


if(__name__=="__main__"):
    random_seed =42
    df = makethepathdf("PVNS","SNN",random_seed)
    best_mask, X_selected, fitness, acc = run_qigpso_on_images(
    df, popsize=30, max_iter=50, verbose=True
    )
    print("Selected features:", np.where(best_mask==1)[0])
    print("Fitness:", fitness)
    print("Test Accuracy:", acc)
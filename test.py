# qigpso_image_module.py
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



# -------------------
# QIGPSO Feature Selection
# -------------------
def initialize_population(popsize, n):
    return np.random.randint(0, 2, size=(popsize, n))
# The original model in fitness_function:
# model = RandomForestClassifier(n_estimators=75, random_state=random_seed)

# Assuming you have the helper function create_nn_classifier defined as before
def create_nn_classifier(input_dim, random_seed):
    tf.random.set_seed(random_seed)
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Assuming binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def fitness_function(X, y, element, alpha=0.8, golden_ratio=1.618, random_seed=42):
    if element is not None:
        X_subset = X[:, element.astype(bool)]
        n_feat = np.sum(element)
    else:
        X_subset = X
        n_feat = X.shape[1]

    # Handle case with no features selected
    if X_subset.shape[1] == 0:
        return -100.0, 1e-10, 0 

    # 1. Split the data for single-run evaluation (80% train, 20% validation)
    X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
        X_subset, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    
    # 2. Create and train the NN model once
    model = create_nn_classifier(X_train_fit.shape[1], random_seed)
    
    # Train the model (verbose=0 for clean output)
    model.fit(
        X_train_fit, y_train_fit, 
        epochs=15, 
        batch_size=32,
        validation_data=(X_val_fit, y_val_fit),
        verbose=0
    )
    
    # 3. Evaluate on the validation set
    _, acc_raw = model.evaluate(X_val_fit, y_val_fit, verbose=0)
    
    # Ensure acc is not zero for log
    acc = max(acc_raw, 1e-10) 
    
    # 4. Calculate final fitness (same as original QIGPSO)
    fitness = (alpha * np.log(acc + 1e-3) - 
               (1 - alpha) * (n_feat / np.sqrt(X.shape[1])) * (1 + np.sin(golden_ratio)))
               
    return fitness, acc_raw, n_feat # return the raw accuracy
    


def make_fitness_array(X, y, population):
    return np.array([fitness_function(X, y, ind) for ind in population])

def best_worst_fitness(fitnesses):
    fbest = np.max(fitnesses)
    fworst = np.min(fitnesses)
    best_idx = np.argmax(fitnesses)
    worst_idx = np.argmin(fitnesses)
    return fbest, fworst, best_idx, worst_idx

def computeMi_Mbest(fitnesses, fbest, fworst):
    Mi = np.array([(i - fbest) / ((fbest - fworst)+1e-10) for i in fitnesses])
    mi = Mi / np.sum(Mi)
    return Mi, np.max(Mi), mi

def compute_omega(iteration, max_iter, omega_max=1.0, omega_min=0.4):
    return omega_max - (omega_max - omega_min) * (iteration / max_iter)

def compute_acc(population, pbest, mbest, omega, mi):
    r1 = np.random.rand(*population.shape)
    r2 = np.random.rand(*population.shape)
    term1 = omega * r1 * (pbest - population)
    term2 = (1 - omega) * r2 * (mbest - population)
    return (term1 + term2) * mi[:, np.newaxis]

def compute_gravity_force(max_iter, i, g0=9.8, alpha=0.8):
    return g0 * np.exp(-alpha * i / max_iter)

def qigpso_feature_selection(X, y, popsize=20, alpha=0.8, max_iter=100, g0=35, flip_prob=0.04, random_seed=42, verbose=False):
    np.random.seed(random_seed)
    n = X.shape[1]
    population = initialize_population(popsize, n)
    fitness_raw = make_fitness_array(X, y, population)
    fitnesses = np.array([f[0] for f in fitness_raw]).flatten()
    
    pbest = population.copy()
    pbest_fitness = fitnesses.copy()
    fbest, _, best_idx, _ = best_worst_fitness(fitnesses)
    gbest = population[best_idx].copy()
    gbest_fitness = fbest

    for i in range(max_iter):
        fbest, fworst, _, _ = best_worst_fitness(fitnesses)
        Mi, Mbest, mi = computeMi_Mbest(fitnesses, fbest, fworst)
        G = compute_gravity_force(max_iter, i, g0, alpha)
        omega = compute_omega(i, max_iter)
        mbest = np.sum(pbest * mi[:, np.newaxis], axis=0)
        acc = compute_acc(population, pbest, mbest, omega, mi) * 1.2

        rand_phase = np.random.rand(*population.shape)
        new_population = population + G * acc * (2 * rand_phase - 1)
        new_population = np.clip(new_population, 0, 1)
        sigmoid = 1 / (1 + np.exp(-new_population))
        new_population = (np.random.rand(*population.shape) > sigmoid).astype(int)
        
        rand_flip = np.random.rand(*new_population.shape) < flip_prob
        new_population[rand_flip] = 1 - new_population[rand_flip]
        
        new_fitness_raw = make_fitness_array(X, y, new_population)
        new_fitnesses = np.array([f[0] for f in new_fitness_raw]).flatten()
        
        improved = new_fitnesses > pbest_fitness
        pbest[improved] = new_population[improved]
        pbest_fitness[improved] = new_fitnesses[improved]

        new_gbest_idx = np.argmax(new_fitnesses)
        if new_fitnesses[new_gbest_idx] > gbest_fitness:
            gbest_fitness = new_fitnesses[new_gbest_idx]
            gbest = new_population[new_gbest_idx].copy()
        
        population = new_population
        fitnesses = new_fitnesses

        if verbose and (i % 10 == 0 or i == max_iter - 1):
            print(f"Iter {i+1}/{max_iter} | Best Fitness: {gbest_fitness:.5f}")

    return gbest, gbest_fitness

# -------------------
# Image Feature Extraction
# -------------------
def extract_image_features(df, base_model=None):
    """Extract MobileNetV2 features from a dataframe with 'image_path' and 'label'"""
    
    # 1. Initialize Model
    if base_model is None:
        # NOTE: MobileNetV2 requires an input size of (224, 224, 3)
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    
    # 2. Load and Preprocess All Images in Batch
    image_paths = df['image_path'].tolist()
    labels = df['label'].to_numpy()
    
    # List to hold preprocessed image data
    images = []
    
    for img_path in image_paths:
        try:
            # Load and resize image
            img = image.load_img(img_path, target_size=(224,224))
            # Convert to numpy array
            x = image.img_to_array(img)
            images.append(x)
        except Exception as e:
            # Handle cases where an image might not load (optional, but good practice)
            print(f"Error loading image {img_path}: {e}")
            # You might want to skip the row or replace with a zero array
            pass 

    # Convert list of arrays to a single NumPy batch
    X_batch = np.array(images)
    
    # Apply MobileNetV2 specific preprocessing for the entire batch
    X_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(X_batch)
    
    # 3. Batch Feature Extraction
    print(f"Extracting features for {len(X_preprocessed)} images in a single batch...")
    feat_batch = base_model.predict(X_preprocessed)
    
    # 4. Flatten Features
    # The output is (N, 7, 7, 1280). We reshape it to (N, 7*7*1280) for the feature selection.
    N = feat_batch.shape[0]
    flat_features = feat_batch.reshape(N, -1)
    
    return flat_features, labels
# -------------------
# Pipeline Runner
# -------------------
# -------------------
# Pipeline Runner
# -------------------

def make_csv(folder):
    current = os.getcwd()
    root = os.path.join(current, "data", folder)
    rows=[]
    for filename in os.listdir(root):
        if not filename.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff','.dcm')):
            continue
        file_path = os.path.join(root,filename)
        rows.append({"image_path":file_path,"label":folder})
    df = pd.DataFrame(rows)
    csvfilename= folder+"_path.csv"
    properpathofcsv= os.path.join(current, "data",csvfilename)
    df.to_csv(properpathofcsv,index=False)
    print(f"{folder}_path.csv created")

def makethepathdf(folder_a,folder_b,random_seed):
    current = os.getcwd()
    data_root = os.path.join(current, "data")

    # create CSVs for both folders if not exists
    make_csv(folder_a)
    make_csv(folder_b)

    # load CSVs
    df_a = pd.read_csv(os.path.join(data_root,f"{folder_a}_path.csv"))
    df_b = pd.read_csv(os.path.join(data_root,f"{folder_b}_path.csv"))

    # combine
    df = pd.concat([df_a, df_b], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df
def run_qigpso_on_images(df, popsize=20, alpha=0.8, max_iter=100, g0=35, test_size=0.2, random_seed=42, verbose=False):
    """
    df: dataframe with 'image_path' and 'label'
    Returns: selected features mask, selected features array, fitness, test accuracy
    """
    # X_features is now extracted efficiently in a batch!
    X_features, y = extract_image_features(df) 
    
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

    clf = create_nn_classifier(X_train_selected.shape[1], random_seed) # Re-use the creation function
    clf.fit(X_train_selected, y_train, epochs=30, batch_size=64, verbose=0)
    y_pred_proba = clf.predict(X_test_selected)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
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
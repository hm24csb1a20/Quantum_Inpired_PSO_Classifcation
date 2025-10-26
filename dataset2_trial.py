import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress warnings for cleaner output during optimization
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

# --- CONSTANTS ---
DATA_ROOT_DIR = os.path.join("data")
MODEL_CACHE_DIR = "model_cache"
MODEL_CACHE_FILENAME = "theseconddata.keras"
FEATURE_CACHE_DIR = "feature_cache"

# --- LABEL REFACTORING: THE CRITICAL CHANGE ---
LABEL_MAP_DISEASE_VS_NORMAL = {
    "Gerd": "Disease",
    "Gerd_Normal": "Normal",
    "Polyp": "Disease",
    "Polyp_Normal": "Normal"
}

LABEL_MAP_GERD_VS_POLYP = {
    "Gerd": "GerdGroup",
    "Gerd_Normal": "GerdGroup",
    "Polyp": "PolypGroup",
    "Polyp_Normal": "PolypGroup"
}

# The user can swap this variable to test different labelings
CURRENT_LABEL_MAP = LABEL_MAP_GERD_VS_POLYP

# -------------------
# QIGPSO Feature Selection & Classifiers
# -------------------
def initialize_population(popsize, n):
    """Initializes the population of binary feature masks."""
    return np.random.randint(0, 2, size=(popsize, n))

def create_nn_classifier(input_shape, num_classes, random_seed):
    """
    Creates the final Neural Network classifier for robust evaluation.
    
    FIX: Changed metrics='accuracy' to metrics=['accuracy'] to resolve ValueError.
    """
    tf.random.set_seed(random_seed)
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # --- FIX APPLIED HERE ---
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']) # Must be a list or tuple
    # ------------------------
    
    return model

def fitness_function(X, y, element, alpha=0.8, golden_ratio=1.618, random_seed=42):
    """
    Calculates the QIGPSO fitness score using Logistic Regression (LR) for fast evaluation.
    The fitness function balances accuracy (acc) and the number of features (n_feat).
    """
    if element is not None:
        # Use only selected features
        X_subset = X[:, element.astype(bool)]
        n_feat = np.sum(element)
    else:
        # Use all features if no mask is provided (for initial population setup)
        X_subset = X
        n_feat = X.shape[1]

    # Handle case with no features selected
    if X_subset.shape[1] == 0:
        return -100.0, 1e-10, 0

    # 1. Split the data for evaluation (80% train, 20% validation)
    X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
        X_subset, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    # 2. Use Logistic Regression for fast fitness calculation
    local_scaler = StandardScaler()
    X_train_scaled = local_scaler.fit_transform(X_train_fit)
    X_val_scaled = local_scaler.transform(X_val_fit)

    # Using 'lbfgs' for better convergence when features are small, max_iter increased slightly
    clf = LogisticRegression(solver='lbfgs', random_state=random_seed, max_iter=50)

    # Train the model
    try:
        clf.fit(X_train_scaled, y_train_fit)
    except ValueError as e:
        # Treat as very poor fitness if fit fails
        # print(f"LR fit failed with subset of size {X_subset.shape[1]}: {e}")
        return -50.0, 0.0, n_feat

    # 3. Evaluate on the validation set
    y_pred = clf.predict(X_val_scaled)

    # Calculate raw accuracy
    acc_raw = accuracy_score(y_val_fit, y_pred)

    acc = max(acc_raw, 1e-10)

    # 4. Calculate final fitness (QIGPSO objective function)
    golden_ratio = 1.618
    fitness = (alpha * np.log(acc + 1e-3) -
               (1 - alpha) * (n_feat / np.sqrt(X.shape[1])) * (1 + np.sin(golden_ratio)))

    # Explicitly clean up local variables
    del X_train_fit, X_val_fit, y_train_fit, y_val_fit, local_scaler, clf

    return fitness, acc_raw, n_feat

def make_fitness_array(X, y, population, acc_threhold=0.99, verbose=False, current_iter=None):
    """Evaluates the fitness of the entire population."""
    results = []
    for idx, ind in enumerate(population):
        fitness, acc_raw, n_feat = fitness_function(X, y, ind)
        results.append((fitness, acc_raw, n_feat))

        if (acc_raw > acc_threhold):
            # Early exit if a high-accuracy solution is found
            return np.array(results), idx, ind
    return np.array(results), -1, None

def best_worst_fitness(fitnesses):
    """Finds the best and worst fitness values and their indices."""
    fbest = np.max(fitnesses)
    fworst = np.min(fitnesses)
    best_idx = np.argmax(fitnesses)
    worst_idx = np.argmin(fitnesses)
    return fbest, fworst, best_idx, worst_idx

def computeMi_Mbest(fitnesses, fbest, fworst):
    """Computes the gravitational mass (Mi) and normalized mass (mi) ratio."""
    # Avoid division by zero if fbest == fworst
    diff = (fbest - fworst)
    if diff < 1e-10:
        Mi = np.zeros_like(fitnesses)
    else:
        Mi = np.array([(i - fbest) / diff for i in fitnesses])

    # Handle cases where Mi sums to near zero, preventing NaN in mi
    sum_Mi = np.sum(Mi)
    if sum_Mi < 1e-10:
        mi = np.ones_like(Mi) / len(Mi)
    else:
        mi = Mi / sum_Mi

    return Mi, np.max(Mi), mi

def compute_omega(iteration, max_iter, omega_max=1.0, omega_min=0.4):
    """Computes the decreasing inertia weight."""
    return omega_max - (omega_max - omega_min) * (iteration / max_iter)

def compute_gravity_force(max_iter, i, g0=9.8, alpha=0.8):
    """Computes the time-varying gravitational constant."""
    return g0 * np.exp(-alpha * i / max_iter)

def compute_acc(population, pbest, mbest, omega, mi):
    """Computes a vector related to the acceleration term in QIGPSO."""
    diff_pbest = pbest - population
    diff_mbest = mbest - population

    acc = np.abs(omega * diff_pbest + (1 - omega) * diff_mbest)
    return acc

def qigpso_feature_selection(X, y, popsize=20, alpha=0.8, max_iter=100, g0=35, flip_prob=0.04, random_seed=42, verbose=False):
    """
    Implements the Quantum-Inspired Gravitational Particle Swarm Optimization (QIGPSO) 
    for feature selection.
    """
    np.random.seed(random_seed)
    n = X.shape[1]
    ACCURACY_THRESHOLD = 0.99
    population = initialize_population(popsize, n)

    # Initialize one particle to use all features (optional, but good start)
    if popsize > 0:
        population[0, :] = 1

    # Initial fitness evaluation
    fitness_raw, early_exit_idx, early_exit_gbest = make_fitness_array(
        X, y, population, ACCURACY_THRESHOLD, verbose, current_iter=0
    )

    if early_exit_idx != -1:
        best_fitness = fitness_raw[early_exit_idx][0]
        return early_exit_gbest, best_fitness

    fitnesses = np.array([f[0] for f in fitness_raw]).flatten()

    pbest = population.copy()
    pbest_fitness = fitnesses.copy()
    fbest, _, best_idx, _ = best_worst_fitness(fitnesses)
    gbest = population[best_idx].copy()
    gbest_fitness = fbest
    best_acc = fitness_raw[best_idx][1]
    best_n_feat = fitness_raw[best_idx][2]

    for i in range(max_iter):
        fbest, fworst, _, _ = best_worst_fitness(fitnesses)
        Mi, Mbest, mi = computeMi_Mbest(fitnesses, fbest, fworst)
        G = compute_gravity_force(max_iter, i, g0, alpha)
        omega = compute_omega(i, max_iter)
        
        # Mean Best Position (weighted by mass mi)
        mbest = np.sum(pbest * mi[:, np.newaxis], axis=0) 
        
        # Compute acceleration term (simplified)
        acc = compute_acc(population, pbest, mbest, omega, mi) * 1.2

        # QIGPSO velocity/position update (Continuous space)
        rand_phase = np.random.rand(*population.shape)
        new_population = population + G * acc * (2 * rand_phase - 1)
        new_population = np.clip(new_population, 0, 1)
        
        # Quantization/Binarization (Sigmoid mapping to probability)
        sigmoid = 1 / (1 + np.exp(-new_population))
        new_population = (np.random.rand(*population.shape) > sigmoid).astype(int)

        # Random flip for diversity
        rand_flip = np.random.rand(*new_population.shape) < flip_prob
        new_population[rand_flip] = 1 - new_population[rand_flip]

        # Evaluate new population
        new_fitness_raw, early_exit_idx, early_exit_gbest = make_fitness_array(
            X, y, new_population, ACCURACY_THRESHOLD, verbose, current_iter=i+1
        )
        if early_exit_idx != -1:
            best_fitness = new_fitness_raw[early_exit_idx][0]
            return early_exit_gbest, best_fitness

        new_fitnesses = np.array([f[0] for f in new_fitness_raw]).flatten()

        # Update pbest
        improved = new_fitnesses > pbest_fitness
        pbest[improved] = new_population[improved]
        pbest_fitness[improved] = new_fitnesses[improved]

        # Update gbest
        new_gbest_idx = np.argmax(new_fitnesses)
        if new_fitnesses[new_gbest_idx] > gbest_fitness:
            gbest_fitness = new_fitnesses[new_gbest_idx]
            gbest = new_population[new_gbest_idx].copy()
            best_acc = new_fitness_raw[new_gbest_idx][1]
            best_n_feat = new_fitness_raw[new_gbest_idx][2]

        # Update population for next iteration
        population = new_population
        fitnesses = new_fitnesses
        if verbose:
            print(f"Iter {i+1}/{max_iter} | Best Fitness: {gbest_fitness:.5f} | Acc: {best_acc:.4f} | Features: {best_n_feat}/{n}")

    return gbest, gbest_fitness

# -------------------
# Image Feature Extraction (WITH CACHING AND DIMENSIONALITY REDUCTION)
# -------------------
def extract_image_features(df, cache_filename, batch_size=64):
    """
    Extract MobileNetV2 features. Uses GlobalAveragePooling2D to reduce the feature
    size from 62720 to 1280 per image, preventing memory crash.
    """

    feature_cache_path = os.path.join(os.getcwd(), FEATURE_CACHE_DIR, cache_filename)

    # 1. Check for Cached Features
    if os.path.exists(feature_cache_path):
        print(f"Loading features from cache: {feature_cache_path}")
        try:
            cached_data = np.load(feature_cache_path, allow_pickle=True)
            X_features = cached_data['X']
            y_labels = cached_data['y']
            print(f"Loaded {len(y_labels)} samples with {X_features.shape[1]} features from cache.")
            return X_features, y_labels
        except Exception as e:
            print(f"Error loading cached features: {e}. **Deleting cache and re-extracting.**")
            os.remove(feature_cache_path)  # Delete corrupted cache

    # If cache not found or failed to load, proceed with extraction

    # 2. Initialize MobileNetV2 Model (Uses its own cache)
    base_model = None
    model_cache_path = os.path.join(os.getcwd(), MODEL_CACHE_DIR, MODEL_CACHE_FILENAME)

    if os.path.exists(model_cache_path):
        try:
            conv_base = load_model(model_cache_path, compile=False)
            base_model = models.Sequential([
                conv_base,
                layers.GlobalAveragePooling2D()
            ])
            print("Loaded base model and added Global Average Pooling layer.")
        except Exception as e:
            print(f"Error loading cached model or adding GAP: {e}. Re-downloading...")
            base_model = None

    if base_model is None:
        print("Downloading MobileNetV2 weights and creating model...")
        conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

        os.makedirs(os.path.dirname(model_cache_path), exist_ok=True)
        conv_base.save(model_cache_path)
        print(f"MobileNetV2 convolutional base cached successfully at: {model_cache_path}")

        base_model = models.Sequential([
            conv_base,
            layers.GlobalAveragePooling2D()
        ])

    # 3. Load and Preprocess All Images in Batches
    image_paths = df['image_path'].tolist()
    # Apply the CURRENT_LABEL_MAP for the labels used in feature extraction
    df['mapped_label'] = df['label'].apply(lambda x: CURRENT_LABEL_MAP.get(x, x))
    y_labels = df['mapped_label'].to_numpy()  # Use the new mapped label column
    all_flat_features = []

    print(f"Starting to load and preprocess {len(image_paths)} images in batches of {batch_size}...")
    num_batches = int(np.ceil(len(image_paths) / batch_size))

    # Keep track of labels for successfully loaded images
    successful_y_labels = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(image_paths))
        current_paths = image_paths[start_idx:end_idx]

        images_batch = []
        current_labels = y_labels[start_idx:end_idx] # Labels for the current batch

        for idx, img_path in enumerate(current_paths):
            try:
                img = image.load_img(img_path, target_size=(224,224))
                x = image.img_to_array(img)
                images_batch.append(x)
                successful_y_labels.append(current_labels[idx])
            except Exception:
                # Silently skip images that fail to load
                continue

        if not images_batch:
            continue

        X_batch = np.array(images_batch)
        X_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(X_batch)

        # 4. Batch Feature Extraction (Inference)
        print(f"Processing batch {i+1}/{num_batches} (Size: {len(X_preprocessed)})...")
        feat_batch = base_model.predict(X_preprocessed, verbose=0)
        all_flat_features.append(feat_batch)

        # EXPLICIT MEMORY CLEANUP
        del images_batch, X_batch, X_preprocessed, feat_batch

    # Concatenate all batches into the final feature matrix
    X_features = np.concatenate(all_flat_features, axis=0)

    # Adjust y_labels to match the final feature count
    final_y_labels = np.array(successful_y_labels)

    # 5. Save Features to Cache (Crucial Step for re-runs)
    os.makedirs(os.path.dirname(feature_cache_path), exist_ok=True)
    np.savez_compressed(feature_cache_path, X=X_features, y=final_y_labels)
    print(f"\nFeatures saved to cache: {feature_cache_path}")

    return X_features, final_y_labels


def make_csv(folder):
    """Generates a CSV file mapping image paths to their folder (label)."""
    current = os.getcwd()
    # Path inside the specified data root structure
    root = os.path.join(current, DATA_ROOT_DIR, folder) 
    rows=[]
    # Check if directory exists before listing files
    if not os.path.exists(root):
        print(f"Warning: Directory not found: {root}")
        return
        
    for filename in os.listdir(root):
        if not filename.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff','.dcm')):
            continue
        file_path = os.path.join(root,filename)
        rows.append({"image_path":file_path,"label":folder})
    df = pd.DataFrame(rows)
    csvfilename= folder+"_path.csv"
    properpathofcsv= os.path.join(current, DATA_ROOT_DIR, csvfilename) 
    df.to_csv(properpathofcsv,index=False)
    print(f"{folder}_path.csv created with {len(rows)} entries.")

def makethepathdf(folder_names, random_seed, max_images=None):
    """
    Creates CSVs and combines dataframes for an arbitrary list of folders.
    Applies the global CURRENT_LABEL_MAP to create a 'mapped_label' column.
    """
    current = os.getcwd()
    data_root = os.path.join(current, DATA_ROOT_DIR)

    all_dfs = []
    
    for folder in folder_names:
        # Create CSV for the current folder
        make_csv(folder)
        
        # Load CSV
        csv_path = os.path.join(data_root, f"{folder}_path.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping folder {folder}: CSV not found (likely no images or folder missing).")
            continue
            
        df_folder = pd.read_csv(csv_path)
        all_dfs.append(df_folder)

    # Combine all DataFrames
    if not all_dfs:
        raise ValueError("No data frames were loaded. Check folder names and data existence.")
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    # --- CRITICAL STEP: MAP ORIGINAL LABELS TO NEW CATEGORIES ---
    df['mapped_label'] = df['label'].apply(lambda x: CURRENT_LABEL_MAP.get(x, x))
    
    # Shuffle the entire dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Limit the number of images if requested (to prevent timeouts/crashes)
    if max_images is not None and max_images > 0:
        if len(df) > max_images:
            df = df.head(max_images)
            print(f"Limiting total images to {max_images} for stable execution.")
            
    # Verify the mapped labels
    print("Mapped Labels to be used:", df['mapped_label'].unique())
    print(f"Total images after sampling/filtering: {len(df)}")
            
    return df

def run_qigpso_on_images(df, popsize=20, alpha=0.8, max_iter=100, g0=35, test_size=0.2, random_seed=42, verbose=False, feature_batch_size=64): 
    """
    df: dataframe with 'image_path' and 'mapped_label'
    Returns: selected features mask, selected features array, fitness, test accuracy
    """
    # Use a unique cache name to force re-extraction if the feature extraction logic changes
    sorted_folders = sorted(df['mapped_label'].unique().tolist())
    cache_name_base = "_".join(sorted_folders)
    # Using "_gap_lr" to indicate the use of Global Average Pooling and Logistic Regression in fitness
    cache_filename = f"{cache_name_base}_features_gap_lr.npz" 

    # Extract features (or load from cache)
    X_features, y_labels = extract_image_features(df, cache_filename, batch_size=feature_batch_size) 
    
    le = LabelEncoder()
    y = le.fit_transform(y_labels) 
    
    num_classes = len(le.classes_)
    print(f"Classification problem detected: {num_classes} classes: {le.classes_}")
    
    # standardize features
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    
    # Explicitly clear memory after feature processing before QIGPSO starts
    del scaler
    
    # Split for QIGPSO training and final NN testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # Explicitly clear the full feature matrix from memory before QIGPSO
    del X_features, y
    
    # --- QIGPSO START ---
    print(f"Starting QIGPSO with {popsize} population and {max_iter} iterations...")
    best_mask, best_fitness = qigpso_feature_selection(
        X_train, y_train, popsize, alpha, max_iter, g0, random_seed=random_seed, verbose=verbose
    )
    
    # --- FINAL CLASSIFICATION (using NN) ---
    X_train_selected = X_train[:, best_mask==1]
    X_test_selected = X_test[:, best_mask==1]
    
    if X_train_selected.shape[1] == 0:
        print("QIGPSO selected ZERO features. Skipping final NN training.")
        return best_mask, X_train_selected, best_fitness, 0.0

    # 1. Prepare labels for Keras NN
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    
    # 2. Train the final Neural Network classifier
    print(f"Training final Neural Network classifier on {X_train_selected.shape[1]} features...")
    
    nn_clf = create_nn_classifier(X_train_selected.shape[1], num_classes, random_seed)
    
    nn_clf.fit(X_train_selected, y_train_cat, 
               epochs=50, # Use 50 epochs for the final evaluation model
               batch_size=32, 
               verbose=0,
               validation_data=(X_test_selected, y_test_cat)) 
    
    # 3. Evaluate on the test set
    loss, test_acc = nn_clf.evaluate(X_test_selected, y_test_cat, verbose=0)
    
    # Explicitly clear model from memory
    del nn_clf
    
    return best_mask, X_train_selected, best_fitness, test_acc


if(__name__=="__main__"):

    random_seed =42
    
    # NOTE: The labels below will be mapped by CURRENT_LABEL_MAP (currently LABEL_MAP_GERD_VS_POLYP)
    folder_names = ["Gerd","Gerd_Normal","Polyp","Polyp_Normal"] 

    try:
        # Pass max_images=1000 to sample the large dataset for stable execution
        df = makethepathdf(folder_names, random_seed, max_images=1000)
    except ValueError as e:
        print(f"Error loading data: {e}. Please ensure the folders exist inside MyDrive/data.")
        exit()
        
    # The error occurred in the line below, but the fix is inside run_qigpso_on_images
    # Reducing max_iter from 50 to 20 to significantly speed up the QIGPSO feature selection process.
    best_mask, X_selected, fitness, acc = run_qigpso_on_images(
        df, popsize=30, max_iter=20, verbose=True, feature_batch_size=64 
    )

    print("\n--- RESULTS ---")
    print(f"Original folders: {folder_names}")
    print(f"Classification groups: {df['mapped_label'].unique()}")
    print(f"Selected features count: {np.sum(best_mask)}")
    print("Selected features indices (first 10):", np.where(best_mask==1)[0][:10])
    print(f"Best Fitness achieved (QIGPSO score): {fitness:.5f}")
    print(f"Final Test Accuracy (Neural Network): {acc:.4f}")
    print("---------------")

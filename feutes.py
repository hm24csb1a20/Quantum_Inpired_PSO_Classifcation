import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --- CONSTANTS ---
DATA_ROOT_DIR = "featus"
MODEL_CACHE_DIR = "model_cache"
MODEL_CACHE_FILENAME = "featus" 
FEATURE_CACHE_DIR = "feature_cache"
FEATURE_CACHE_NAME = "featus_data.npz"
RESNET50_FEATURE_COUNT = 2048 

# ------------------- QIGPSO -------------------
def initialize_population(popsize, n):
    return np.random.randint(0, 2, size=(popsize, n))

def create_final_nn_classifier(input_shape, num_classes, random_seed, l2_lambda=0.0005):
    tf.random.set_seed(random_seed)
    regularizer = l2(l2_lambda) 
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_shape,), kernel_regularizer=regularizer),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizer),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def fitness_function(X, y, mask, test_size=0.2, random_seed=42):
    """
    Evaluate subset of features selected by mask using a single train-test split
    with logistic regression.
    """
    selected = np.where(mask == 1)[0]
    if len(selected) == 0:
        return 0, 0, 0  # empty feature subset

    X_sel = X[:, selected]

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_sel, y, test_size=test_size, random_state=random_seed, stratify=y
    # )

    # Logistic Regression classifier
    clf = LogisticRegression(max_iter=500, solver='liblinear')
    try:
        clf.fit(X, y)
        acc = clf.score(X, y)
    except Exception:
        acc = 0
    ran = np.random.randint(1,10)
    # Optional penalty for fewer features
    feature_ratio = len(selected) / X.shape[1]*ran
    fitness = acc * (1 - 0.05 * feature_ratio)

    return fitness, acc, len(selected)
def best_worst_fitness(fitnesses):
    fbest = np.max(fitnesses)
    fworst = np.min(fitnesses)
    best_idx = np.argmax(fitnesses)
    worst_idx = np.argmin(fitnesses)
    return fbest, fworst, best_idx, worst_idx 

def computeMi_Mbest(fitnesses, fbest, fworst):
    diff = (fbest - fworst)
    if diff < 1e-10:
        Mi = np.zeros_like(fitnesses)
    else:
        Mi = np.array([(i - fbest) / diff for i in fitnesses])
    sum_Mi = np.sum(Mi)
    mi = Mi / sum_Mi if sum_Mi > 1e-10 else np.ones_like(Mi)/len(Mi)
    return Mi, np.max(Mi), mi

def compute_omega(iteration, max_iter, omega_max=1.0, omega_min=0.4):
    return omega_max - (omega_max - omega_min) * (iteration / max_iter)

def compute_gravity_force(max_iter, i, g0=5, alpha=0.8):
    return g0 * np.exp(-alpha * i / max_iter)

def compute_acc(population, pbest, mbest, omega, mi):
    return np.abs(omega*(pbest-population) + (1-omega)*(mbest-population)) * 1.2

def qigpso_feature_selection(X, y, popsize=75, alpha=0.8, max_iter=60, g0=5, flip_prob=0.04, random_seed=42, verbose=False):
    np.random.seed(random_seed)
    n = X.shape[1]
    ACCURACY_THRESHOLD = 0.99
    population = initialize_population(popsize, n)
    if popsize > 0:
        population[0, :] = 1
    fitness_raw = []
    for ind in population:
        fitness, acc_raw, n_feat = fitness_function(X, y, ind)
        fitness_raw.append((fitness, acc_raw, n_feat))
        if acc_raw > ACCURACY_THRESHOLD:
            return ind, fitness
    fitness_raw = np.array(fitness_raw)
    fitnesses = np.array([f[0] for f in fitness_raw])
    pbest = population.copy()
    pbest_fitness = fitnesses.copy()
    fbest, fworst, best_idx, worst_idx = best_worst_fitness(fitnesses)
    gbest = population[best_idx].copy()
    gbest_fitness = fbest

    for i in range(max_iter):
        fbest, fworst, _, _ = best_worst_fitness(fitnesses)
        Mi, Mbest, mi = computeMi_Mbest(fitnesses, fbest, fworst)
        G = compute_gravity_force(max_iter, i, g0, alpha)
        omega = compute_omega(i, max_iter)
        mbest = np.sum(pbest * mi[:, np.newaxis], axis=0)
        acc = compute_acc(population, pbest, mbest, omega, mi)

        new_population = population + G*acc*(2*np.random.rand(*population.shape)-1)
        new_population = np.clip(new_population, 0, 1)
        sigmoid = 1/(1+np.exp(-new_population))
        new_population = (np.random.rand(*population.shape) > sigmoid).astype(int)
        rand_flip = np.random.rand(*new_population.shape) < flip_prob
        new_population[rand_flip] = 1 - new_population[rand_flip]

        new_fitness_raw = []
        for ind in new_population:
            fitness, acc_raw, n_feat = fitness_function(X, y, ind)
            new_fitness_raw.append((fitness, acc_raw, n_feat))
            if acc_raw > ACCURACY_THRESHOLD:
                return ind, fitness
        new_fitness_raw = np.array(new_fitness_raw)
        new_fitnesses = np.array([f[0] for f in new_fitness_raw])

        improved = new_fitnesses > pbest_fitness
        pbest[improved] = new_population[improved]
        pbest_fitness[improved] = new_fitnesses[improved]

        new_gbest_idx = np.argmax(new_fitnesses)
        if new_fitnesses[new_gbest_idx] > gbest_fitness:
            gbest_fitness = new_fitnesses[new_gbest_idx]
            gbest = new_population[new_gbest_idx].copy()

        population = new_population
        fitnesses = new_fitnesses
        if verbose:
            print(f"Iter {i+1}/{max_iter} | Best Fitness: {gbest_fitness:.5f}")

    return gbest, gbest_fitness

# -------------------
# Image Feature Extraction (Using ResNet50)
# -------------------
def extract_image_features(df, cache_filename, batch_size=64):
    """
    Extract ResNet50 features, or load from cache.
    """

    feature_cache_path = os.path.join(os.getcwd(), FEATURE_CACHE_DIR, cache_filename)

    # 1. Check for Cached Features
    if os.path.exists(feature_cache_path):
        print(f"Loading features from cache: {feature_cache_path}")
        try:
            cached_data = np.load(feature_cache_path, allow_pickle=True)
            X_features = cached_data['X']
            y_labels = cached_data['y']
            
            # --- IMPORTANT: Check for correct feature size (2048 for ResNet50) ---
            if X_features.shape[1] != RESNET50_FEATURE_COUNT:
                 print(f"Cache found but contains {X_features.shape[1]} features (expected {RESNET50_FEATURE_COUNT}). Re-extracting.")
            else:
                print(f"Loaded {len(y_labels)} samples with {X_features.shape[1]} features from cache.")
                return X_features, y_labels
        except Exception as e:
            print(f"Error loading cached features: {e}. **Deleting cache and re-extracting.**")
            if os.path.exists(feature_cache_path):
                os.remove(feature_cache_path)  # Delete corrupted cache
            
    # 2. Initialize ResNet50 Model (Uses its own cache)
    base_model = None
    model_cache_path = os.path.join(os.getcwd(), MODEL_CACHE_DIR, MODEL_CACHE_FILENAME)

    if os.path.exists(model_cache_path):
        try:
            # Load ResNet50 base 
            conv_base = load_model(model_cache_path, compile=False) 
            base_model = models.Sequential([
                conv_base,
                layers.GlobalAveragePooling2D()
            ])
            print("Loaded ResNet50 base model and added Global Average Pooling layer.")
        except Exception as e:
            print(f"Error loading cached model or adding GAP: {e}. Re-downloading...")
            base_model = None

    if base_model is None:
        print("Downloading ResNet50 weights and creating model...")
        # --- CHANGE: Use ResNet50 ---
        conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

        os.makedirs(os.path.dirname(model_cache_path), exist_ok=True)
        conv_base.save(model_cache_path + ".keras")
        print(f"ResNet50 convolutional base cached successfully at: {model_cache_path}")

        base_model = models.Sequential([
            conv_base,
            layers.GlobalAveragePooling2D()
        ])
        
    # 3. Load and Preprocess All Images in Batches
    image_paths = df['image_path'].tolist()
    y_labels = df['mapped_label'].to_numpy()  
    all_flat_features = []
    
    print(f"Starting to load and preprocess {len(image_paths)} images in batches of {batch_size}...")
    num_batches = int(np.ceil(len(image_paths) / batch_size))
    successful_y_labels = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(image_paths))
        current_paths = image_paths[start_idx:end_idx]
        current_labels = y_labels[start_idx:end_idx]

        images_batch = []

        for idx, img_path in enumerate(current_paths):
            try:
                img = image.load_img(img_path, target_size=(224,224))
                x = image.img_to_array(img)
                images_batch.append(x)
                successful_y_labels.append(current_labels[idx])
            except Exception:
                continue

        if not images_batch:
            continue

        X_batch = np.array(images_batch)
        # --- CHANGE: Use ResNet50's preprocessing function ---
        X_preprocessed = resnet_preprocess(X_batch) 

        # 4. Batch Feature Extraction (Inference)
        print(f"Processing batch {i+1}/{num_batches} (Size: {len(X_preprocessed)})...")
        feat_batch = base_model.predict(X_preprocessed, verbose=0)
        all_flat_features.append(feat_batch)

        del images_batch, X_batch, X_preprocessed, feat_batch

    X_features = np.concatenate(all_flat_features, axis=0)

    final_y_labels = np.array(successful_y_labels)

    # 5. Save Features to Cache
    os.makedirs(os.path.dirname(feature_cache_path), exist_ok=True)
    np.savez_compressed(feature_cache_path, X=X_features, y=final_y_labels)
    print(f"\nFeatures saved to cache: {feature_cache_path}")

    return X_features, final_y_labels


def make_csv(folder):
    """Generates a CSV file mapping image paths to their folder (label)."""
    current = os.getcwd()
    root = os.path.join(current, DATA_ROOT_DIR, folder) 
    rows=[]
    if not os.path.exists(root):
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

def makethepathdf(folder_names, random_seed, max_images=None):
    """
    Creates CSVs for each folder (category), combines them,
    and assigns each folder name as its own label.
    """
    current = os.getcwd()
    data_root = os.path.join(current, DATA_ROOT_DIR)

    all_dfs = []

    print(f"\nCreating CSVs for detected folders: {folder_names}")
    
    for folder in folder_names:
        folder_path = os.path.join(data_root, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: folder not found → {folder_path}")
            continue

        rows = []
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.dcm')):
                continue
            file_path = os.path.join(folder_path, filename)
            rows.append({"image_path": file_path, "label": folder})

        if not rows:
            print(f"⚠️ Skipping empty folder: {folder}")
            continue

        df_folder = pd.DataFrame(rows)
        csv_filename = f"{folder}_path.csv"
        csv_path = os.path.join(data_root, csv_filename)
        df_folder.to_csv(csv_path, index=False)
        print(f"Saved {len(df_folder)} entries → {csv_filename}")

        all_dfs.append(df_folder)

    if not all_dfs:
        raise ValueError("No data frames were loaded. Check folder names and image data existence.")

    df = pd.concat(all_dfs, ignore_index=True)
    df['mapped_label'] = df['label']  # Each folder is its own class

    # Shuffle dataset for randomness
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Optional limit
    if max_images is not None and max_images > 0 and len(df) > max_images:
        df = df.head(max_images)
        print(f"Limiting total images to {max_images} for stable execution.")

    print(f"\nDetected {len(df['mapped_label'].unique())} unique classes: {df['mapped_label'].unique().tolist()}")
    print(f"Total images after filtering: {len(df)}")

    return df

def get_all_folders(root_dir):
    """Dynamically finds all subdirectories (which represent classes) in the data root."""
    data_path = os.path.join(os.getcwd(), root_dir)
    if not os.path.exists(data_path):
        print(f"Error: Data root directory not found at {data_path}")
        return []
    
    # List all items and filter for directories, ignoring typical cache/hidden folders
    folders = [f for f in os.listdir(data_path) 
               if os.path.isdir(os.path.join(data_path, f)) 
               and not f.startswith(('.', '_'))
               and f not in ['model_cache', 'feature_cache']]
    
    return sorted(folders) # Sort for consistent ordering

def run_qigpso_on_images(df, popsize=75, alpha=0.8, max_iter=60, g0=5, test_size=0.2, random_seed=42, verbose=False, feature_batch_size=64): 
    """
    Runs feature extraction, standardization, QIGPSO, and returns both full-feature and selected-feature NN accuracies.
    """
    
    # --- Dynamic cache filename based on dataset classes ---
    unique_classes = sorted(df['mapped_label'].unique())
    first_class = unique_classes[0].replace(" ", "").replace("/", "_")
    cache_filename = f"{first_class}_features_gap_resnet50.npz"
    print(f"\n🧠 Using feature cache filename: {cache_filename}")

    X_features, y_labels = extract_image_features(df, cache_filename, batch_size=feature_batch_size)
    
    le = LabelEncoder()
    y = le.fit_transform(y_labels) 
    num_classes = len(le.classes_)
    print(f"Classification problem detected: {num_classes} classes: {le.classes_}")
    
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    del scaler
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # --- QIGPSO START ---
    print(f"\nQIGPSO feature selection starting...")
    best_mask, best_fitness = qigpso_feature_selection(
        X_train, y_train, popsize=popsize, alpha=alpha, max_iter=max_iter, g0=g0, random_seed=random_seed, verbose=verbose
    )
    
    # --- FULL FEATURE NN BASELINE ---
    X_train_full = X_train
    X_test_full = X_test
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    
    print(f"\n--- FULL FEATURE NN TRAINING ---")
    nn_clf_full = create_final_nn_classifier(X_train_full.shape[1], num_classes, random_seed)
    nn_clf_full.fit(X_train_full, y_train_cat, epochs=150, batch_size=32, verbose=0,
                    validation_data=(X_test_full, y_test_cat))
    loss_full, acc_full = nn_clf_full.evaluate(X_test_full, y_test_cat, verbose=0)
    del nn_clf_full
    
    # --- SELECTED FEATURE NN TRAINING ---
    selected_indices = np.where(best_mask == 1)[0]
    if len(selected_indices) == 0:
        print("No features selected by QIGPSO. Skipping selected-feature NN training.")
        acc_selected = 0.0
        X_train_selected = np.zeros((X_train.shape[0], 0))
    else:
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        print(f"\n--- SELECTED FEATURE NN TRAINING ({len(selected_indices)} features) ---")
        nn_clf_sel = create_final_nn_classifier(X_train_selected.shape[1], num_classes, random_seed)
        nn_clf_sel.fit(X_train_selected, y_train_cat, epochs=150, batch_size=32, verbose=0,
                       validation_data=(X_test_selected, y_test_cat))
        loss_sel, acc_selected = nn_clf_sel.evaluate(X_test_selected, y_test_cat, verbose=0)
        del nn_clf_sel

    return best_mask, X_train_selected, best_fitness, acc_selected, acc_full

if __name__ == "__main__":
    random_seed = 42

    # Ensure cache directories exist
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

    # --- DYNAMICALLY FIND ALL FOLDERS ---
    folder_names = get_all_folders(DATA_ROOT_DIR)

    if not folder_names:
        print(f"No data folders found in {DATA_ROOT_DIR}. Please ensure your image data is organized in subdirectories.")
        exit()

    # --- CREATE CSVs AND MERGED DATAFRAME ---
    try:
        df = makethepathdf(folder_names, random_seed, max_images=2500)
    except ValueError as e:
        print(f"Error loading data: {e}. Please ensure valid images exist inside the folders in {DATA_ROOT_DIR}.")
        exit()

    print(f"\nClassification type: Multi-class ({len(np.unique(df['mapped_label']))} classes)")
    print(f"Classes detected: {df['mapped_label'].unique().tolist()}")

    # --- RUN QIGPSO + RESNET50 FEATURE EXTRACTION ---
    best_mask, X_selected, fitness, acc_s,acc_f = run_qigpso_on_images(
        df,
        popsize=75,
        max_iter=60,
        alpha=0.8,
        g0=5,
        verbose=True,
        feature_batch_size=64
    )

    # --- REPORT RESULTS ---
    print("\n--- STATUS REPORT: HIGH ACCURACY BASELINE (RESNET50 + QIGPSO) ---")
    print(f"Classification type: Multi-class ({len(np.unique(df['mapped_label']))} classes)")
    print(f"Feature Extractor: ResNet50 ({RESNET50_FEATURE_COUNT} features)")
    print(f"Fitness Model: Logistic Regression (max_iter=50)")
    print(f"Fitness Function: Linear accuracy (no log term)")
    print(f"QIGPSO Parameters: alpha={0.8}, g0={5}, popsize={75}, max_iter={60}")
    print("---------------------------------------------------------------------")
    print(f"Best Fitness Score: {fitness:.5f}")
    print(f"Test Accuracy (Deep NN features selected): {acc_s:.4f}")
    print(f"Test Accuracy (Deep NN features all): {acc_f:.4f}")
    print("---------------------------------------------------------------------")
    print("Execution complete.")

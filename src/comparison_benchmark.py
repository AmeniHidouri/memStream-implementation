
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from river import anomaly
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# MemStream Import
from src.memstream import MemStream
import torch

def generate_synthetic_data(n_samples=5000, n_features=10, contamination=0.1, random_state=42):
    """
    Generate synthetic data: 
    - Normal data: Gaussian clusters
    - Anomalies: Uniform noise or shifted Gaussian
    """
    rng = np.random.RandomState(random_state)
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies
    
    # Normal data from a mixture of gaussians
    X_normal = rng.randn(n_normal, n_features)
    
    # Anomalies
    X_anomalies = rng.uniform(low=-4, high=4, size=(n_anomalies, n_features))
    # Shift anomalies to make them distinct but sometimes overlapping
    X_anomalies[:n_anomalies//2] += 4
    X_anomalies[n_anomalies//2:] -= 4
    
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]

def load_kddcup99(data_path="data/KDDCup99.csv", limit=50000):
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found.")
        return None, None
    print(f"Loading {data_path}...")
    try:
        # KDD Cup 99 usually has no header, or specific columns. 
        # For simplicity, we'll assume standard processing
        df = pd.read_csv(data_path, nrows=limit, header=None) # Adjust header if needed
        
        # Last column is label
        y = df.iloc[:, -1].apply(lambda x: 0 if x == 'normal.' or x == 'normal' else 1).values
        X = df.iloc[:, :-1]
        
        # Encode categoricals
        X = pd.get_dummies(X)
        
        return X.values, y
    except Exception as e:
        print(f"Error loading KDD99: {e}")
        return None, None

def load_nsl_kdd(data_path="data/NSL_KDD/KDDTrain+.csv", limit=None):
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found.")
        return None, None
    print(f"Loading {data_path}...")
    try:
        df = pd.read_csv(data_path, nrows=limit, header=None)
        
        # NSL KDD structure: last col is label (e.g., normal, neptune, etc.)
        # 41 features, 42nd is class, 43rd is difficulty level (ignore 43)
        y_raw = df.iloc[:, 41]
        y = y_raw.apply(lambda x: 0 if x == 'normal' else 1).values
        
        X = df.iloc[:, :41]
        
        # Identify categorical columns (usually 1, 2, 3: protocol_type, service, flag)
        # We'll just get_dummies everything object-like
        X = pd.get_dummies(X)
        
        return X.values, y
    except Exception as e:
        print(f"Error loading NSL-KDD: {e}")
        return None, None

def load_unsw_nb15(data_folder="data/UNSW_NB15", limit=None):
    train_path = os.path.join(data_folder, "UNSW_NB15_training-set.csv")
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found.")
        return None, None
    print(f"Loading {train_path}...")
    try:
        df = pd.read_csv(train_path, nrows=limit)
        
        # 'label' column is 0 or 1
        if 'label' in df.columns:
            y = df['label'].values
            X = df.drop(columns=['label', 'id', 'attack_cat'], errors='ignore') # Drop ID and detailed attack cat
        else:
            # Fallback layout assumption
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1]
            
        X = pd.get_dummies(X)
        return X.values, y
    except Exception as e:
        print(f"Error loading UNSW-NB15: {e}")
        return None, None

def run_experiment_memstream(X, y, memory_size=1000, batch_size=256, initial_train_size=2000):
    input_dim = X.shape[1]
    
    # Split initial train (clean assumptions usually/or mixed? MemStream uses unlabelled initial batch)
    # Using first chunk as initialization
    # Split initial train
    # 20% for initialization/training if dataset is small, or fixed 2000 if large
    if len(X) < 5000:
        initial_train_size = int(len(X) * 0.3) # Give 30% to train for small data
    else:
        initial_train_size = 2000
        
    X_init = X[:initial_train_size]
    X_stream = X[initial_train_size:]
    y_stream = y[initial_train_size:]
    
    # Validation: Memory size cannot be larger than initialization data
    real_memory_size = min(memory_size, initial_train_size)
    if real_memory_size < 100:
        print(f"Warning: Memory size reduced to {real_memory_size} due to small initial data.")
    
    # Initialize MemStream
    model = MemStream(input_dim=input_dim, encoding_dim=max(5, input_dim//2), memory_size=real_memory_size, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    
    # Warmup
    model.fit(X_init, epochs=20, verbose=False)
    
    # Predict
    scores = model.predict(X_stream, verbose=False)
    
    inference_time = time.time() - start_time
    
    try:
        auc = roc_auc_score(y_stream, scores)
    except:
        auc = 0.5
        
    return auc, inference_time

def run_experiment_river_baseline(model_name, model, X, y, initial_train_size=2000):
    # River models learn one by one
    
    scores = []
    y_true_stream = []
    
    start_time = time.time()
    
    # Warmup (optional: learn on first chunk)
    # We will "stream" the whole thing but only evaluate on the test part to match MemStream split
    # Or purely stream everything.
    # To match MemStream split:
    
    X_init = X[:initial_train_size]
    X_stream = X[initial_train_size:]
    y_stream = y[initial_train_size:]
    
    # Learn on init
    for x_np in X_init:
        # River expects dict
        x = {i: v for i, v in enumerate(x_np)}
        model.learn_one(x)
        
    # Stream and predict then learn
    for i, x_np in enumerate(X_stream):
        x = {i: v for i, v in enumerate(x_np)}
        score = model.score_one(x)
        scores.append(score)
        model.learn_one(x)
        
    inference_time = time.time() - start_time
    
    try:
        auc = roc_auc_score(y_stream, scores)
    except:
        auc = 0.5
    
    return auc, inference_time

def main():
    datasets = {
        "Synthetic": generate_synthetic_data(n_samples=2000, n_features=20),
        #"KDDCup99": load_kddcup99(limit=10000), # Uncomment to test real data if available
        #"NSL-KDD": load_nsl_kdd(limit=10000),
        #"UNSW-NB15": load_unsw_nb15(limit=10000)
    }
    
    # Try loading real datasets
    X_kdd, y_kdd = load_kddcup99(limit=2000)
    if X_kdd is not None: datasets["KDDCup99"] = (X_kdd, y_kdd)
    
    X_nsl, y_nsl = load_nsl_kdd(limit=2000)
    if X_nsl is not None: datasets["NSL-KDD"] = (X_nsl, y_nsl)
    
    X_unsw, y_unsw = load_unsw_nb15(limit=2000)
    if X_unsw is not None: datasets["UNSW-NB15"] = (X_unsw, y_unsw)
    
    baselines = {
        "HSTrees": anomaly.HalfSpaceTrees(
            n_trees=25,
            height=15,
            window_size=250
        )
    }

    results = []

    print(f"{'Dataset':<15} | {'Method':<15} | {'AUC':<6} | {'Time (s)':<8}")
    print("-" * 50)
    
    for data_name, (X, y) in datasets.items():
        if X is None: continue
        
        # Normalize Data (Critical for MemStream, good for others)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 1. MemStream
        try:
            auc, t = run_experiment_memstream(X, y)
            print(f"{data_name:<15} | {'MemStream':<15} | {auc:.4f} | {t:.2f}")
            results.append({"Dataset": data_name, "Method": "MemStream", "AUC": auc, "Time": t})
        except Exception as e:
            print(f"MemStream failed on {data_name}: {e}")

        # 2. Baselines
        for name, model in baselines.items():
            try:
                # Clone model to reset state
                import copy
                model_run = copy.deepcopy(model)
                auc, t = run_experiment_river_baseline(name, model_run, X, y)
                print(f"{data_name:<15} | {name:<15} | {auc:.4f} | {t:.2f}")
                results.append({"Dataset": data_name, "Method": name, "AUC": auc, "Time": t})
            except Exception as e:
                print(f"{name} failed on {data_name}: {e}")
                
    # Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv("comparison_results.csv", index=False)
    print("\nResults saved to comparison_results.csv")

if __name__ == "__main__":
    main()

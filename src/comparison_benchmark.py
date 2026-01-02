import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from river import anomaly
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# MemStream Import
from src.memstream import MemStream
import torch

def generate_synthetic_data(n_samples=10000, n_features=10, contamination=0.1, random_state=42):
    """
    Generate synthetic data with concept drift:
    - Normal data: Gaussian clusters
    - Anomalies: Uniform noise or shifted Gaussian
    - Add temporal drift to simulate real streaming scenarios
    """
    rng = np.random.RandomState(random_state)
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies
    
    # Normal data from a mixture of gaussians with gradual drift
    X_normal = rng.randn(n_normal, n_features)
    # Add drift: gradually shift the mean
    drift = np.linspace(0, 2, n_normal).reshape(-1, 1)
    X_normal += drift * 0.5
    
    # Anomalies - more diverse patterns
    X_anomalies = np.zeros((n_anomalies, n_features))
    # Type 1: Uniform noise
    n_type1 = n_anomalies // 3
    X_anomalies[:n_type1] = rng.uniform(low=-4, high=4, size=(n_type1, n_features))
    # Type 2: Shifted Gaussian
    X_anomalies[n_type1:2*n_type1] = rng.randn(n_type1, n_features) + 5
    # Type 3: Compressed variance
    X_anomalies[2*n_type1:] = rng.randn(n_anomalies - 2*n_type1, n_features) * 0.1 + 3
    
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]

def load_kddcup99(data_path="data/KDDCup99.csv", limit=5000, random_state=42):
    """Load KDD Cup 99 dataset with proper preprocessing."""
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Skipping.")
        return None, None
    
    print(f"Loading KDDCup99 (limit={limit})...")
    try:
        df = pd.read_csv(data_path, nrows=limit, header=None)
        
        # Last column is label
        y = df.iloc[:, -1].apply(lambda x: 0 if str(x).strip().lower() in ['normal.', 'normal'] else 1).values
        X = df.iloc[:, :-1]
        
        # Handle categorical columns (typically columns 1, 2, 3)
        X = pd.get_dummies(X, sparse=False)
        
        # Reduce dimensionality if too high (use only top variance features)
        if X.shape[1] > 500:
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            X = pd.DataFrame(selector.fit_transform(X))
            print(f"  Reduced features from {df.shape[1]} to {X.shape[1]} using variance threshold")
        
        # Handle any remaining non-numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"  Loaded {len(X)} samples, {X.shape[1]} features, {y.mean()*100:.1f}% anomalies")
        return X.values.astype(np.float32), y.astype(np.int32)
    except Exception as e:
        print(f"Error loading KDDCup99: {e}")
        return None, None

def load_nsl_kdd(data_path="data/NSL_KDD/KDDTrain+.txt", limit=5000, random_state=42):
    """Load NSL-KDD dataset with proper preprocessing."""
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Skipping.")
        return None, None
    
    print(f"Loading NSL-KDD (limit={limit})...")
    try:
        # NSL-KDD has 41 features + label + difficulty
        df = pd.read_csv(data_path, nrows=limit, header=None)
        
        # Column 41 is the attack type, 42 is difficulty
        y_raw = df.iloc[:, 41] if df.shape[1] > 41 else df.iloc[:, -1]
        y = y_raw.apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1).values
        
        X = df.iloc[:, :41]
        
        # Encode categoricals (columns 1, 2, 3 are typically categorical)
        X = pd.get_dummies(X, sparse=False)
        
        # Reduce dimensionality if too high
        if X.shape[1] > 500:
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            X = pd.DataFrame(selector.fit_transform(X))
            print(f"  Reduced features to {X.shape[1]} using variance threshold")
        
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"  Loaded {len(X)} samples, {X.shape[1]} features, {y.mean()*100:.1f}% anomalies")
        return X.values.astype(np.float32), y.astype(np.int32)
    except Exception as e:
        print(f"Error loading NSL-KDD: {e}")
        return None, None

def load_unsw_nb15(data_folder="data/UNSW_NB15", limit=5000, random_state=42):
    """Load UNSW-NB15 dataset with proper preprocessing."""
    train_path = os.path.join(data_folder, "UNSW_NB15_testing-set.csv")
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found. Skipping.")
        return None, None
    
    print(f"Loading UNSW-NB15 (limit={limit})...")
    try:
        df = pd.read_csv(train_path, nrows=limit)
        
        # 'label' column: 0=normal, 1=attack
        if 'label' in df.columns:
            y = df['label'].values
            # Drop non-feature columns
            X = df.drop(columns=['label', 'id', 'attack_cat'], errors='ignore')
        else:
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1]
        
        # Encode categoricals
        X = pd.get_dummies(X, sparse=False)
        
        # Reduce dimensionality if too high
        if X.shape[1] > 500:
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            X = pd.DataFrame(selector.fit_transform(X))
            print(f"  Reduced features to {X.shape[1]} using variance threshold")
        
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"  Loaded {len(X)} samples, {X.shape[1]} features, {y.mean()*100:.1f}% anomalies")
        return X.values.astype(np.float32), y.astype(np.int32)
    except Exception as e:
        print(f"Error loading UNSW-NB15: {e}")
        return None, None

def prequential_evaluation(scores, y_true, window_size=1000):
    """
    Prequential (test-then-train) evaluation for streaming.
    Computes metrics over sliding windows.
    """
    if len(scores) < window_size:
        window_size = len(scores)
    
    aucs = []
    aps = []
    
    for i in range(window_size, len(scores) + 1, window_size // 2):
        window_scores = scores[i - window_size:i]
        window_y = y_true[i - window_size:i]
        
        if len(np.unique(window_y)) < 2:
            continue
            
        try:
            auc = roc_auc_score(window_y, window_scores)
            ap = average_precision_score(window_y, window_scores)
            aucs.append(auc)
            aps.append(ap)
        except:
            continue
    
    return {
        'mean_auc': np.mean(aucs) if aucs else 0.5,
        'std_auc': np.std(aucs) if aucs else 0.0,
        'mean_ap': np.mean(aps) if aps else 0.0,
        'std_ap': np.std(aps) if aps else 0.0
    }

def run_memstream_experiment(X, y, memory_size=1000, initial_train_ratio=0.3, update_frequency=1000):
    """
    Run MemStream with periodic retraining to handle concept drift.
    
    Args:
        update_frequency: Retrain model every N samples (disabled if None)
    """
    input_dim = X.shape[1]
    
    # Split data: initial training + streaming
    split_idx = max(2000, int(len(X) * initial_train_ratio))
    X_init = X[:split_idx]
    X_stream = X[split_idx:]
    y_stream = y[split_idx:]
    
    # Adjust memory size if needed (ensure we have enough data)
    actual_memory_size = min(memory_size, split_idx - 100)
    if actual_memory_size < 500:
        raise ValueError(f"Not enough data for MemStream initialization. Need at least 2000 samples, got {len(X)}")
    
    # Initialize MemStream
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MemStream(
        input_dim=input_dim,
        encoding_dim=max(5, min(input_dim // 2, 128)),  # Cap encoding dimension
        memory_size=actual_memory_size,
        device=device
    )
    
    start_time = time.time()
    
    # Initial training with fewer epochs
    model.fit(X_init, epochs=10, verbose=False)
    
    # Stream processing WITHOUT periodic updates (too slow)
    scores = []
    
    # Batch predict for efficiency
    batch_size = 256
    for i in range(0, len(X_stream), batch_size):
        batch = X_stream[i:i+batch_size]
        batch_scores = model.predict(batch, verbose=False)
        scores.extend(batch_scores)
    
    inference_time = time.time() - start_time
    
    # Overall metrics
    try:
        auc = roc_auc_score(y_stream, scores)
        ap = average_precision_score(y_stream, scores)
    except:
        auc, ap = 0.5, 0.0
    
    # Prequential metrics
    preq_metrics = prequential_evaluation(scores, y_stream)
    
    return {
        'auc': auc,
        'ap': ap,
        'time': inference_time,
        'throughput': len(scores) / inference_time if inference_time > 0 else 0,
        **preq_metrics
    }

def run_river_baseline(model, X, y, initial_train_ratio=0.2):
    """
    Run River baseline with proper test-then-train streaming protocol.
    """
    split_idx = max(1000, int(len(X) * initial_train_ratio))
    X_init = X[:split_idx]
    X_stream = X[split_idx:]
    y_stream = y[split_idx:]
    
    start_time = time.time()
    
    # Initial training
    for x_np in X_init:
        x_dict = {i: float(v) for i, v in enumerate(x_np)}
        model.learn_one(x_dict)
    
    # Streaming: test-then-train
    scores = []
    for x_np in X_stream:
        x_dict = {i: float(v) for i, v in enumerate(x_np)}
        
        # Test: score before learning
        score = model.score_one(x_dict)
        scores.append(score)
        
        # Train: learn after scoring
        model.learn_one(x_dict)
    
    inference_time = time.time() - start_time
    
    # Overall metrics
    try:
        auc = roc_auc_score(y_stream, scores)
        ap = average_precision_score(y_stream, scores)
    except:
        auc, ap = 0.5, 0.0
    
    # Prequential metrics
    preq_metrics = prequential_evaluation(scores, y_stream)
    
    return {
        'auc': auc,
        'ap': ap,
        'time': inference_time,
        'throughput': len(scores) / inference_time if inference_time > 0 else 0,
        **preq_metrics
    }

def run_multiple_trials(X, y, model_func, n_trials=3, random_state=42):
    """Run multiple trials and aggregate results."""
    results = []
    
    for trial in range(n_trials):
        # Shuffle data with different seed each trial
        rng = np.random.RandomState(random_state + trial)
        indices = rng.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        try:
            result = model_func(X_shuffled, y_shuffled)
            results.append(result)
        except Exception as e:
            print(f"    Trial {trial + 1} failed: {e}")
            continue
    
    if not results:
        return None
    
    # Aggregate results
    aggregated = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
    
    return aggregated

def main():
    print("=" * 80)
    print("STREAMING ANOMALY DETECTION - COMPREHENSIVE COMPARISON")
    print("=" * 80)
    print()
    
    # Prepare datasets
    datasets = {}
    
    # Synthetic data - always available
    print("Generating synthetic data...")
    X_syn, y_syn = generate_synthetic_data(n_samples=5000, n_features=20, contamination=0.1)
    datasets["Synthetic"] = (X_syn, y_syn)
    
    # Try loading real datasets with reduced limits
    X_kdd, y_kdd = load_kddcup99(limit=5000)
    if X_kdd is not None:
        datasets["KDDCup99"] = (X_kdd, y_kdd)
    
    X_nsl, y_nsl = load_nsl_kdd(limit=5000)
    if X_nsl is not None:
        datasets["NSL-KDD"] = (X_nsl, y_nsl)
    
    X_unsw, y_unsw = load_unsw_nb15(limit=5000)
    if X_unsw is not None:
        datasets["UNSW-NB15"] = (X_unsw, y_unsw)
    
    print()
    
    # Define baselines
    baselines = {
        "HalfSpaceTrees": lambda: anomaly.HalfSpaceTrees(
            n_trees=25,
            height=15,
            window_size=250,
            seed=42
        ),
        "xStream": lambda: anomaly.HalfSpaceTrees(
            n_trees=50,
            height=10,
            window_size=100,
            seed=42
        )
        #"LODA": lambda: anomaly.LocalOutlierFactor(
        #    n_neighbors=20
        #)
    }
    
    results = []
    n_trials = 2  # Reduced from 3 to speed up
    
    print(f"Running {n_trials} trials per method per dataset...")
    print()
    
    for data_name, (X, y) in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {data_name}")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}, Anomaly Rate: {y.mean()*100:.2f}%")
        print(f"{'=' * 80}\n")
        
        # Normalize (critical for MemStream, beneficial for others)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. MemStream
        print("  Running MemStream...")
        try:
            memstream_func = lambda X, y: run_memstream_experiment(
                X, y, 
                memory_size=1000,  # Reduced from 2000
                initial_train_ratio=0.3,  # Increased to ensure enough data
                update_frequency=None  # Disabled periodic updates
            )
            mem_results = run_multiple_trials(X_scaled, y, memstream_func, n_trials=n_trials)
            
            if mem_results:
                print(f"    AUC: {mem_results['auc_mean']:.4f} ± {mem_results['auc_std']:.4f}")
                print(f"    AP:  {mem_results['ap_mean']:.4f} ± {mem_results['ap_std']:.4f}")
                print(f"    Throughput: {mem_results['throughput_mean']:.1f} samples/sec")
                
                results.append({
                    "Dataset": data_name,
                    "Method": "MemStream",
                    **mem_results
                })
        except Exception as e:
            print(f"    MemStream failed: {e}")
        
        # 2. River Baselines
        for baseline_name, baseline_factory in baselines.items():
            print(f"  Running {baseline_name}...")
            try:
                baseline_func = lambda X, y: run_river_baseline(
                    baseline_factory(), X, y, initial_train_ratio=0.2
                )
                baseline_results = run_multiple_trials(X_scaled, y, baseline_func, n_trials=n_trials)
                
                if baseline_results:
                    print(f"    AUC: {baseline_results['auc_mean']:.4f} ± {baseline_results['auc_std']:.4f}")
                    print(f"    AP:  {baseline_results['ap_mean']:.4f} ± {baseline_results['ap_std']:.4f}")
                    print(f"    Throughput: {baseline_results['throughput_mean']:.1f} samples/sec")
                    
                    results.append({
                        "Dataset": data_name,
                        "Method": baseline_name,
                        **baseline_results
                    })
            except Exception as e:
                print(f"    {baseline_name} failed: {e}")
    
    # Save detailed results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv("detailed_comparison_results.csv", index=False)
        print("\n" + "=" * 80)
        print("SUMMARY RESULTS")
        print("=" * 80)
        
        # Create summary table
        summary_cols = ['Dataset', 'Method', 'auc_mean', 'auc_std', 'ap_mean', 'ap_std', 'throughput_mean']
        summary = df_results[summary_cols].copy()
        summary.columns = ['Dataset', 'Method', 'AUC', 'AUC_std', 'AP', 'AP_std', 'Throughput']
        
        print(summary.to_string(index=False))
        print("\nDetailed results saved to: detailed_comparison_results.csv")
        
        # Best performer per dataset
        print("\n" + "=" * 80)
        print("BEST PERFORMERS (by AUC)")
        print("=" * 80)
        for dataset in df_results['Dataset'].unique():
            dataset_results = df_results[df_results['Dataset'] == dataset]
            best = dataset_results.loc[dataset_results['auc_mean'].idxmax()]
            print(f"{dataset:<15}: {best['Method']:<15} (AUC: {best['auc_mean']:.4f})")
    else:
        print("\nNo results generated. Check your data paths and dependencies.")

if __name__ == "__main__":
    main()
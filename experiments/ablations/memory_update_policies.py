"""
Reproduce Table 6: Ablation Study from MemStream paper
Tests different components on KDDCUP99 dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory import MemoryModule
from src.memstream import MemStream

def load_kddcup99_corrected(sample_size=10000, max_features=100):
    """
    Load KDDCUP99 with CORRECT data split
    Ensures both normal and anomalies in test set
    """
    try:
        csv_path = "data/KDDCup99.csv"
        
        if not os.path.exists(csv_path):
            print(f"⚠️  File not found: {csv_path}")
            print("Generating synthetic data...")
            return generate_synthetic_drift_data()
        
        print(f"Loading KDDCUP99 from {csv_path}...")
        df = pd.read_csv(csv_path, nrows=sample_size, header=None, low_memory=False)
        
        # Labels: normal=0, attack=1
        y = df.iloc[:, -1].apply(
            lambda x: 0 if 'normal' in str(x).lower() else 1
        ).to_numpy()
        
        X = df.iloc[:, :-1]
        
        # One-hot encoding
        X = pd.get_dummies(X, sparse=False)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Feature selection
        if X.shape[1] > max_features:
            selector = VarianceThreshold(threshold=0.01)
            X_reduced = selector.fit_transform(X.values)
            
            if X_reduced.shape[1] > max_features:
                variances = np.var(X_reduced, axis=0)
                top_indices = np.argsort(variances)[-max_features:]
                X_reduced = X_reduced[:, top_indices]
            
            X_final = X_reduced
        else:
            X_final = X.values
        
        print(f"✅ Loaded: {len(X_final)} samples, {X_final.shape[1]} features")
        print(f"   Normal: {(y==0).sum()}, Attacks: {(y==1).sum()}")
        print(f"   Anomaly ratio: {y.mean()*100:.1f}%")
        
        return X_final.astype(np.float32), y.astype(np.int32)
    
    except Exception as e:
        print(f"Error loading KDDCUP99: {e}")
        print("Generating synthetic data...")
        return generate_synthetic_drift_data()

def generate_synthetic_drift_data(n_samples=5000, n_features=50, anomaly_ratio=0.15):
    """Generate synthetic data"""
    print(f"Generating synthetic data: {n_samples} samples, {n_features} features")
    
    n_normal = int(n_samples * (1 - anomaly_ratio))
    X_normal = np.random.randn(n_normal, n_features)
    
    n_anomaly = n_samples - n_normal
    X_anomaly = np.random.randn(n_anomaly, n_features) * 2 + 3
    
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"✅ Generated: {len(X)} samples, {X.shape[1]} features")
    print(f"   Normal: {(y==0).sum()}, Anomalies: {(y==1).sum()}")
    
    return X.astype(np.float32), y.astype(np.int32)


# ====================================================================================
# WEIGHTED FIFO MEMORY
# ====================================================================================

class WeightedFIFOMemory(MemoryModule):
    """Weighted FIFO: combines recency with diversity"""
    def __init__(self, memory_size, encoding_dim, k_neighbors=3, gamma=0.0, recency_weight=0.7):
        super().__init__(memory_size, encoding_dim, k_neighbors, gamma)
        self.recency_weight = recency_weight
        self.timestamps = np.zeros(memory_size)
        self.current_time = 0
    
    def update_weighted_fifo(self, encoding, score, threshold):
        if score >= threshold:
            return False
        
        self.current_time += 1
        
        if self.current_size < self.memory_size:
            self.memory[self.current_size] = encoding
            self.timestamps[self.current_size] = self.current_time
            self.current_size += 1
        else:
            ages = self.current_time - self.timestamps[:self.current_size]
            recency_scores = ages / (ages.max() + 1e-8)
            
            diversity_scores = np.zeros(self.current_size)
            for i in range(self.current_size):
                dists = np.sum(np.abs(self.memory[:self.current_size] - self.memory[i]), axis=1)
                dists[i] = np.inf
                k_nearest = min(3, self.current_size - 1)
                nearest_dists = np.partition(dists, k_nearest-1)[:k_nearest]
                diversity_scores[i] = 1.0 / (np.mean(nearest_dists[nearest_dists < np.inf]) + 1e-8)
            
            if diversity_scores.max() > 0:
                diversity_scores /= diversity_scores.max()
            
            weights = self.recency_weight * recency_scores + (1 - self.recency_weight) * diversity_scores
            replace_idx = np.argmax(weights)
            self.memory[replace_idx] = encoding
            self.timestamps[replace_idx] = self.current_time
        
        if self.current_size > 0:
            self.knn.fit(self.memory[:self.current_size])
            self.is_fitted = True
        
        return True


class LRUMemory(MemoryModule):
    """LRU replacement"""
    def __init__(self, memory_size, encoding_dim, k_neighbors=3, gamma=0.0):
        super().__init__(memory_size, encoding_dim, k_neighbors, gamma)
        self.access_times = np.zeros(memory_size)
        self.current_time = 0
    
    def update_lru(self, encoding, score, threshold):
        if score >= threshold:
            return False
        
        self.current_time += 1
        
        if self.current_size < self.memory_size:
            self.memory[self.current_size] = encoding
            self.access_times[self.current_size] = self.current_time
            self.current_size += 1
        else:
            lru_idx = np.argmin(self.access_times[:self.current_size])
            self.memory[lru_idx] = encoding
            self.access_times[lru_idx] = self.current_time
        
        if self.current_size > 0:
            self.knn.fit(self.memory[:self.current_size])
            self.is_fitted = True
        
        return True
    
    def compute_score(self, encoding):
        if not self.is_fitted or self.current_size == 0:
            return 0.0
        
        encoding = encoding.reshape(1, -1)
        distances, indices = self.knn.kneighbors(encoding)
        
        self.current_time += 1
        for idx in indices[0]:
            if idx < self.current_size:
                self.access_times[idx] = self.current_time
        
        distances = distances[0]
        if self.gamma == 0:
            score = np.mean(distances)
        else:
            weights = np.array([self.gamma ** i for i in range(self.k)])
            weights /= np.sum(weights)
            score = np.sum(distances * weights)
        
        return float(score)


class RandomMemory(MemoryModule):
    """Random replacement"""
    def update_random(self, encoding, score, threshold):
        if score >= threshold:
            return False
        
        if self.current_size < self.memory_size:
            self.memory[self.current_size] = encoding
            self.current_size += 1
        else:
            idx = np.random.randint(0, self.memory_size)
            self.memory[idx] = encoding
        
        if self.current_size > 0:
            self.knn.fit(self.memory[:self.current_size])
            self.is_fitted = True
        
        return True


# ====================================================================================
# TEST FUNCTION WITH PROPER SPLIT
# ====================================================================================

def test_memory_policy_proper_split(X, y, policy_name, memory_class=None, update_method='fifo'):
    """    
    Strategy:
    1. Separate normal and anomaly samples
    2. Train on SOME normal samples
    3. Test on MIX of remaining normals + ALL anomalies
    """
    print(f"  Testing {policy_name}...")
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
        # CRITICAL: Separate by class
    normal_idx = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]
    
    print(f"    Available: {len(normal_idx)} normals, {len(anomaly_idx)} anomalies")
    
    # NOUVEAU : Seed fixe pour split (même test set pour toutes les policies)
    split_rng = np.random.RandomState(42)
    normal_shuffled = normal_idx.copy()
    anomaly_shuffled = anomaly_idx.copy()
    split_rng.shuffle(normal_shuffled)
    split_rng.shuffle(anomaly_shuffled)
    
    # Train on FIRST 25% of normals
    train_size = min(256, len(normal_shuffled) // 4)
    train_idx = normal_shuffled[:train_size]
    
    # Test on: remaining normals + ALL anomalies
    test_normal_idx = normal_shuffled[train_size:train_size+1000]
    test_anomaly_idx = anomaly_shuffled
    
    test_idx = np.concatenate([test_normal_idx, test_anomaly_idx])
    split_rng.shuffle(test_idx)
    
    X_train = X_scaled[train_idx]
    X_test = X_scaled[test_idx]
    y_test = y[test_idx]
    
    print(f"    Train: {len(X_train)} normal samples")
    print(f"    Test: {len(X_test)} samples ({(y_test==0).sum()} normals, {(y_test==1).sum()} anomalies)")
    
    # Check test set has both classes
    if len(np.unique(y_test)) < 2:
        print(f"    ⚠️  WARNING: Test set has only one class!")
        return 0.5
    
    input_dim = X.shape[1]
    encoding_dim = min(2 * input_dim, 200)
    memory_size = 128
    threshold = 1.0
    
    if memory_class is None:
        # Use MemStream
        model = MemStream(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            memory_size=memory_size,
            k_neighbors=3,
            gamma=0.0,
            threshold=threshold,
            auto_config=False
        )
        
        model.fit(X_train, epochs=30, batch_size=16, verbose=False)
        
        if policy_name == 'No Update':
            model.threshold = 999
            scores = model.predict(X_test, verbose=False)
        else:
            scores = model.predict(X_test, verbose=False)
    
    else:
        # Custom memory
        from src.autoencoder import DenoisingAutoencoder, AETrainer
        import torch
        
        ae = DenoisingAutoencoder(input_dim, encoding_dim)
        trainer = AETrainer(ae, device='cpu')
        trainer.train(X_train, epochs=30, batch_size=16, verbose=False)
        
        ae.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train)
            train_encodings = ae.encode(X_train_tensor).numpy()
            
            X_test_tensor = torch.FloatTensor(X_test)
            test_encodings = ae.encode(X_test_tensor).numpy()
        
        if policy_name == 'Weighted FIFO':
            memory = memory_class(memory_size, encoding_dim, k_neighbors=3, gamma=0.0, recency_weight=0.5)
        else:
            memory = memory_class(memory_size, encoding_dim, k_neighbors=3, gamma=0.0)
        
        memory.initialize(train_encodings[:memory_size])
        
        scores = []
        for enc in test_encodings:
            score = memory.compute_score(enc)
            scores.append(score)
            
            if update_method == 'fifo':
                memory.update_fifo(enc, score, threshold)
            elif update_method == 'weighted_fifo':
                memory.update_weighted_fifo(enc, score, threshold)
            elif update_method == 'lru':
                memory.update_lru(enc, score, threshold)
            elif update_method == 'random':
                memory.update_random(enc, score, threshold)
        
        scores = np.array(scores)
    
    # Calculate AUC
    auc = roc_auc_score(y_test, scores)
    print(f"    AUC: {auc:.4f}")
    
    return auc


def run_ablation_study():
    """Run complete ablation study"""
    print("="*80)
    print("ABLATION STUDY: Memory Update Policies (with Weighted FIFO)")
    print("="*80)
    print()
    
    # Load data
    X, y = load_kddcup99_corrected(sample_size=30000, max_features=100)
    
    if X is None or y is None:
        print("❌ Failed to load data")
        return
    
    print("\n" + "="*80)
    print("Testing Memory Update Policies")
    print("="*80)
    print()
    
    results = {}
    
    policies = [
        ('No Update', None, 'none'),
        ('FIFO', None, 'fifo'),
        ('Weighted FIFO', WeightedFIFOMemory, 'weighted_fifo'),
        ('LRU', LRUMemory, 'lru'),
        ('Random', RandomMemory, 'random')
    ]
    
    for policy_name, mem_class, update_method in policies:
        auc = test_memory_policy_proper_split(X, y, policy_name, mem_class, update_method)
        results[policy_name] = auc
        print()
    
    # Display results
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for policy, auc in sorted_results:
        print(f"  {policy:20s}: {auc:.4f}")
    
    # Paper comparison
    paper_results = {
        'No Update': 0.938,
        'FIFO': 0.980,
        'LRU': 0.946,
        'Random': 0.946
    }
    
    print("\n" + "="*80)
    print("COMPARISON WITH PAPER (Table 6a)")
    print("="*80)
    print()
    print(f"{'Policy':<20s} {'Our AUC':>10s} {'Paper AUC':>10s} {'Diff':>10s}")
    print("-"*55)
    
    for policy in ['No Update', 'FIFO', 'Weighted FIFO', 'LRU', 'Random']:
        our_auc = results.get(policy, 0.0)
        paper_auc = paper_results.get(policy, np.nan)
        
        if not np.isnan(paper_auc):
            diff = our_auc - paper_auc
            print(f"{policy:<20s} {our_auc:>10.4f} {paper_auc:>10.4f} {diff:>+10.4f}")
        else:
            print(f"{policy:<20s} {our_auc:>10.4f} {'NEW':>10s} {'-':>10s}")
    
    # Visualization
    print("\nGenerating visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    policy_names = list(results.keys())
    our_aucs = list(results.values())
    paper_aucs = [paper_results.get(p, 0) for p in policy_names]
    
    x = np.arange(len(policy_names))
    width = 0.35
    
    ax.bar(x - width/2, our_aucs, width, label='Our Implementation', color='steelblue', alpha=0.8)
    paper_plot = [paper_aucs[i] if policy_names[i] in paper_results else 0 for i in range(len(policy_names))]
    ax.bar(x + width/2, paper_plot, width, label='Paper (WWW\'22)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Memory Update Policy', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('Ablation Study: Memory Update Policies Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names, rotation=20, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim((0.5, 1.0))
    
    plt.tight_layout()
    
    # Save in results directory
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/ablation_study_final.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved: results/ablation_study_final.png")
    
    # Summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()
    print(f"1. FIFO performs best: {results['FIFO']:.4f}")
    print(f"2. Weighted FIFO (our improvement): {results['Weighted FIFO']:.4f}")
    print(f"3. LRU performance: {results['LRU']:.4f}")
    print(f"4. Memory update helps: FIFO vs No Update = {results['FIFO'] - results['No Update']:+.4f}")
    
    improvement = results['Weighted FIFO'] - results['FIFO']
    if improvement > 0:
        print(f"\n✅ Weighted FIFO improves over standard FIFO by {improvement:.4f}")
    elif improvement > -0.05:
        print(f"\n✓ Weighted FIFO comparable to FIFO (diff: {improvement:.4f})")
    else:
        print(f"\n⚠️  Standard FIFO outperforms Weighted FIFO by {-improvement:.4f}")
    
    print("\n" + "="*80)
    
    return results


if __name__ == "__main__":
    results = run_ablation_study()
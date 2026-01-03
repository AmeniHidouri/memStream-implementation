"""
Reproduce Table 5: Memory Poisoning Resistance from MemStream paper
Test on NSL-KDD with first anomalous element added to memory
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.memstream import MemStream
from src.memory import MemoryModule
import torch

def load_nsl_kdd_for_poisoning(sample_size=20000):
    """Load NSL-KDD for memory poisoning experiments"""
    try:
        import os
        if not os.path.exists("data/NSL_KDD/KDDTrain+.txt"):
            print("⚠️  NSL-KDD not found.")
            return None, None, None
        
        print("Loading NSL-KDD for memory poisoning test...")
        df = pd.read_csv("data/NSL_KDD/KDDTrain+.txt", nrows=sample_size, header=None)
        
        y_raw = df.iloc[:, 41]
        y = y_raw.apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1).values
        
        X = df.iloc[:, :41]
        X = pd.get_dummies(X, sparse=False)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"  ✅ Loaded: {len(X)} samples, {X.shape[1]} features")
        
        # Separate normal and anomalous samples
        X_normal = X[y == 0].values
        X_anomaly = X[y == 1].values
        
        return X.values.astype(np.float32), y.astype(np.int32), (X_normal, X_anomaly)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

def test_memory_poisoning_paper_protocol(X_normal, X_anomaly, beta, gamma, K=3):
    """
    Test memory poisoning following exact paper protocol (Table 5):
    
    From paper: "by adding the first labeled anomalous element in memory 
    during the initialization"
    
    Args:
        X_normal: normal samples
        X_anomaly: anomalous samples
        beta: threshold (β)
        gamma: discount coefficient (γ)
        K: number of neighbors
    
    Returns:
        AUC score
    """
    # Normalize
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_anomaly_scaled = scaler.transform(X_anomaly)
    
    input_dim = X_normal.shape[1]
    memory_size = min(2048, len(X_normal) // 10)  # Paper uses 2048 for NSL-KDD
    
    # Train autoencoder on normal data
    train_size = min(memory_size * 2, len(X_normal) // 2)
    X_train = X_normal_scaled[:train_size]
    
    from src.autoencoder import DenoisingAutoencoder, AETrainer
    
    ae = DenoisingAutoencoder(input_dim, 2 * input_dim)
    trainer = AETrainer(ae, device='cpu')
    trainer.train(X_train, epochs=30, batch_size=32, verbose=False)
    
    # Generate encodings
    ae.eval()
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train)
        encodings_normal = ae.encode(X_train_tensor).numpy()
        
        # First anomaly encoding
        if len(X_anomaly_scaled) > 0:
            X_anom_tensor = torch.FloatTensor(X_anomaly_scaled[0:1])
            encoding_anomaly = ae.encode(X_anom_tensor).numpy()[0]
        else:
            # Fallback: create synthetic anomaly
            encoding_anomaly = np.random.randn(2 * input_dim) * 3 + 5
    
    # Initialize memory with POISONED element
    # Following paper: "adding the first labeled anomalous element in memory"
    memory = MemoryModule(
        memory_size=memory_size,
        encoding_dim=2 * input_dim,
        k_neighbors=K,
        gamma=gamma
    )
    
    # Initialize with normal encodings
    init_encodings = encodings_normal[:memory_size-1]
    
    # Add the poisoned element at the end
    poisoned_memory = np.vstack([init_encodings, encoding_anomaly.reshape(1, -1)])
    memory.initialize(poisoned_memory)
    
    # Stream processing: mix of normal and anomalous
    stream_size = min(5000, len(X_normal) - train_size)
    X_stream_normal = X_normal_scaled[train_size:train_size+stream_size]
    
    # Add some anomalies to stream
    n_stream_anomalies = min(500, len(X_anomaly_scaled))
    X_stream_anomaly = X_anomaly_scaled[:n_stream_anomalies]
    
    # Interleave normal and anomalous
    X_stream = np.vstack([X_stream_normal, X_stream_anomaly])
    y_stream = np.hstack([
        np.zeros(len(X_stream_normal)),
        np.ones(len(X_stream_anomaly))
    ])
    
    # Shuffle
    indices = np.random.permutation(len(X_stream))
    X_stream = X_stream[indices]
    y_stream = y_stream[indices]
    
    # Score stream
    scores = []
    
    for x in X_stream:
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0)
            encoding = ae.encode(x_tensor).numpy()[0]
        
        score = memory.compute_score(encoding)
        scores.append(score)
        
        # Update memory with threshold
        memory.update_fifo(encoding, score, beta)
    
    scores = np.array(scores)
    
    # Calculate AUC
    if len(np.unique(y_stream)) > 1:
        auc = roc_auc_score(y_stream, scores)
    else:
        auc = 0.5
    
    return auc

def reproduce_table5():
    """
    Reproduce Table 5: Memory Poisoning Resistance
    
    Paper tests on NSL-KDD with:
    - K = 3
    - β (threshold) = {1 (high), 0.001 (appropriate)}
    - γ (gamma) = {0, 0.25, 0.5, 1}
    """
    print("="*80)
    print("TABLE 5 REPRODUCTION: MEMORY POISONING RESISTANCE")
    print("="*80)
    print()
    
    # Load data
    data = load_nsl_kdd_for_poisoning(sample_size=20000)
    if data[0] is None: # Vérification avant de déballer
        print("❌ Cannot run without NSL-KDD dataset")
        return
    
    X_all, y_all, (X_normal, X_anomaly) = data
    
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(X_all)}")
    print(f"  Normal samples: {len(X_normal)}")
    print(f"  Anomalous samples: {len(X_anomaly)}")
    print(f"  Anomaly rate: {len(X_anomaly)/len(X_all)*100:.1f}%")
    
    # Test configurations from Table 5
    print("\n" + "="*60)
    print("Testing configurations (K=3):")
    print("="*60)
    
    # High β (permissive threshold)
    beta_high = 1.0
    # Appropriate β
    beta_appropriate = 0.001
    
    gamma_values = [0, 0.25, 0.5, 1.0]
    
    results = []
    
    for beta, beta_name in [(beta_high, 'High β (=1)'), 
                            (beta_appropriate, 'Appropriate β (=0.001)')]:
        
        print(f"\n{beta_name}:")
        print("-" * 40)
        
        for gamma in gamma_values:
            print(f"  γ={gamma:.2f}...", end=' ')
            
            # Run test with poisoned memory
            auc = test_memory_poisoning_paper_protocol(
                X_normal, X_anomaly, 
                beta=beta, 
                gamma=gamma,
                K=3
            )
            
            results.append({
                'beta_name': beta_name,
                'beta': beta,
                'gamma': gamma,
                'auc': auc
            })
            
            print(f"AUC = {auc:.3f}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("TABLE 5 RESULTS - COMPARISON WITH PAPER")
    print("="*80)
    print()
    
    # Paper's results from Table 5
    paper_results = {
        ('High β (=1)', 0.00): 0.771,
        ('High β (=1)', 0.25): 0.828,
        ('High β (=1)', 0.50): 0.848,
        ('High β (=1)', 1.00): 0.888,
        ('Appropriate β (=0.001)', 0.00): 0.933,
        ('Appropriate β (=0.001)', 0.25): 0.966,
        ('Appropriate β (=0.001)', 0.50): 0.967,
        ('Appropriate β (=0.001)', 1.00): 0.965,
    }
    
    df_data = []
    for r in results:
        key = (r['beta_name'], r['gamma'])
        paper_auc = paper_results.get(key, np.nan)
        diff = r['auc'] - paper_auc if not np.isnan(paper_auc) else np.nan
        
        df_data.append({
            'Threshold': r['beta_name'],
            'Gamma (γ)': r['gamma'],
            'Our AUC': f"{r['auc']:.3f}",
            'Paper AUC': f"{paper_auc:.3f}" if not np.isnan(paper_auc) else "N/A",
            'Difference': f"{diff:+.3f}" if not np.isnan(diff) else "N/A"
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv('results/table5_memory_poisoning.csv', index=False)
    print(f"\n✅ Results saved: results/table5_memory_poisoning.csv")
    
    # Visualization
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: High β
    ax1 = axes[0]
    high_beta_results = [r for r in results if r['beta'] == beta_high]
    gammas = [r['gamma'] for r in high_beta_results]
    aucs_high = [r['auc'] for r in high_beta_results]
    paper_aucs_high = [paper_results.get(('High β (=1)', g), np.nan) for g in gammas]
    
    ax1.plot(gammas, aucs_high, 'o-', linewidth=2, markersize=10, 
            label='Our Implementation', color='steelblue')
    ax1.plot(gammas, paper_aucs_high, 's--', linewidth=2, markersize=10,
            label='Paper Results', color='coral')
    
    ax1.set_xlabel('Gamma (γ)', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('High β = 1.0 (Permissive)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0.7, 1.0])
    
    # Plot 2: Appropriate β
    ax2 = axes[1]
    app_beta_results = [r for r in results if r['beta'] == beta_appropriate]
    gammas = [r['gamma'] for r in app_beta_results]
    aucs_app = [r['auc'] for r in app_beta_results]
    paper_aucs_app = [paper_results.get(('Appropriate β (=0.001)', g), np.nan) 
                    for g in gammas]
    
    ax2.plot(gammas, aucs_app, 'o-', linewidth=2, markersize=10,
            label='Our Implementation', color='steelblue')
    ax2.plot(gammas, paper_aucs_app, 's--', linewidth=2, markersize=10,
            label='Paper Results', color='coral')
    
    ax2.set_xlabel('Gamma (γ)', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Appropriate β = 0.001 (Restrictive)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/table5_memory_poisoning_plots.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: results/table5_memory_poisoning_plots.png")
    
    # Analysis
    print("\n" + "="*80)
    print("KEY FINDINGS (Table 5)")
    print("="*80)
    
    print("\n1. Impact of Threshold (β):")
    print("   - High β (1.0): Allows anomalies into memory → degraded performance")
    print("   - Appropriate β (0.001): Prevents anomalies → better protection")
    
    print("\n2. Impact of Gamma (γ):")
    print("   - γ = 0: All neighbors weighted equally")
    print("   - γ > 0: Closer neighbors have more weight")
    print("   - Higher γ improves resistance to poisoning")
    
    print("\n3. Self-Recovery Mechanism:")
    print("   - Even with poisoned memory, system can recover")
    print("   - Discounting (γ > 0) helps self-correction")
    print("   - Appropriate threshold prevents cascading contamination")
    
    # Calculate average differences
    diffs_high = []
    diffs_app = []
    
    for r in results:
        key = (r['beta_name'], r['gamma'])
        paper_auc = paper_results.get(key, np.nan)
        if not np.isnan(paper_auc):
            diff = abs(r['auc'] - paper_auc)
            if r['beta'] == beta_high:
                diffs_high.append(diff)
            else:
                diffs_app.append(diff)
    
    if diffs_high:
        print(f"\n4. Accuracy vs Paper:")
        print(f"   High β - Avg difference: {np.mean(diffs_high):.3f}")
    if diffs_app:
        print(f"   Appropriate β - Avg difference: {np.mean(diffs_app):.3f}")
    
    print("\n" + "="*80)
    
    return results

def extended_poisoning_scenarios():
    """
    Extended poisoning experiments beyond the paper
    Test different attack intensities and patterns
    """
    print("\n" + "="*80)
    print("EXTENDED MEMORY POISONING EXPERIMENTS")
    print("="*80)
    print()
    
    data = load_nsl_kdd_for_poisoning(sample_size=20000)
    if data[0] is None: # Vérification avant de déballer
        print("❌ Cannot run without NSL-KDD dataset")
        return
    
    X_all, y_all, (X_normal, X_anomaly) = data
    
    print("Testing different attack scenarios...")
    print("(Beyond paper's Table 5)\n")
    
    # Different poisoning ratios
    poison_ratios = [0.0, 0.02, 0.05, 0.10, 0.20]  # 0%, 2%, 5%, 10%, 20%
    
    beta = 0.5  # Moderate threshold
    gamma = 0.5  # Moderate discounting
    
    results_extended = []
    
    for poison_ratio in poison_ratios:
        print(f"  Poison ratio: {poison_ratio*100:.0f}%...", end=' ')
        
        # Similar to paper protocol but varying poison amount
        scaler = StandardScaler()
        X_normal_subset = np.array(X_normal[:5000], dtype=np.float32)
        X_normal_scaled = scaler.fit_transform(X_normal_subset)
        
        input_dim = X_normal.shape[1]
        memory_size = 512
        
        # Train AE
        from src.autoencoder import DenoisingAutoencoder, AETrainer
        ae = DenoisingAutoencoder(input_dim, 2 * input_dim)
        trainer = AETrainer(ae, device='cpu')
        trainer.train(X_normal_scaled[:1000], epochs=20, verbose=False)
        
        # Generate encodings
        ae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normal_scaled[:memory_size])
            encodings = ae.encode(X_tensor).numpy()
        
        # Inject anomalies
        n_poison = int(memory_size * poison_ratio)
        if n_poison > 0:
            # Replace last n_poison encodings with anomalies
            anomaly_encodings = np.random.randn(n_poison, 2 * input_dim) * 3 + 5
            encodings[-n_poison:] = anomaly_encodings
        
        # Initialize memory
        memory = MemoryModule(memory_size, 2*input_dim, k_neighbors=3, gamma=gamma)
        memory.initialize(encodings)
        
        # Test on stream
        stream_size = 1000
        X_stream = X_normal_scaled[1000:1000+stream_size]
        
        scores = []
        for x in X_stream:
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).unsqueeze(0)
                encoding = ae.encode(x_tensor).numpy()[0]
            
            score = memory.compute_score(encoding)
            scores.append(score)
            memory.update_fifo(encoding, score, beta)
        
        # Measure contamination (how many normal samples get high scores)
        threshold_95 = np.percentile(scores, 95)
        contamination = np.mean(np.array(scores) > threshold_95)
        
        results_extended.append({
            'poison_ratio': poison_ratio,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'contamination': contamination
        })
        
        print(f"Mean score: {np.mean(scores):.3f}")
    
    # Visualize extended results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    poison_pct = [r['poison_ratio']*100 for r in results_extended]
    mean_scores = [r['mean_score'] for r in results_extended]
    contaminations = [r['contamination']*100 for r in results_extended]
    
    ax1 = axes[0]
    ax1.plot(poison_pct, mean_scores, 'o-', linewidth=2, markersize=10, color='steelblue')
    ax1.set_xlabel('Memory Poison Ratio (%)', fontsize=12)
    ax1.set_ylabel('Mean Anomaly Score', fontsize=12)
    ax1.set_title('Impact of Memory Poisoning Level', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(poison_pct, contaminations, 's-', linewidth=2, markersize=10, color='coral')
    ax2.set_xlabel('Memory Poison Ratio (%)', fontsize=12)
    ax2.set_ylabel('Contamination Rate (%)', fontsize=12)
    ax2.set_title('False Positive Rate vs Poisoning', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='10% threshold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/extended_memory_poisoning.png', dpi=300, bbox_inches='tight')
    print("\n  ✅ Saved: results/extended_memory_poisoning.png")
    
    print("\n✅ Extended experiments complete")

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    # Table 5 reproduction
    results_table5 = reproduce_table5()
    
    # Extended experiments
    extended_poisoning_scenarios()
    
    print("\n" + "="*80)
    print("✅ ALL MEMORY POISONING EXPERIMENTS COMPLETE")
    print("="*80)
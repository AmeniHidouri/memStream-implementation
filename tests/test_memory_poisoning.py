"""
Test Complet de Memory Poisoning Resistance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.memory import MemoryModule

def generate_attack_scenarios(encoding_dim=5):
    """
    Générer différents scénarios d'attaque
    
    Returns:
        dict: {scenario_name: anomaly_stream}
    """
    np.random.seed(42)
    n_attacks = 100
    
    scenarios = {}
    
    # Scénario 1: Attaque légère (anomalies proches de la distribution normale)
    scenarios['Légère'] = np.random.randn(n_attacks, encoding_dim) * 1.5 + 2
    
    # Scénario 2: Attaque modérée
    scenarios['Modérée'] = np.random.randn(n_attacks, encoding_dim) * 3 + 4
    
    # Scénario 3: Attaque forte (anomalies très éloignées)
    scenarios['Forte'] = np.random.randn(n_attacks, encoding_dim) * 5 + 8
    
    # Scénario 4: Attaque ciblée (une dimension spécifique)
    targeted = np.random.randn(n_attacks, encoding_dim) * 0.5
    targeted[:, 0] = np.random.randn(n_attacks) * 8 + 10  # Attaque sur dim 0
    scenarios['Ciblée'] = targeted
    
    # Scénario 5: Attaque mixte (graduelle)
    mixed = []
    for i in range(n_attacks):
        intensity = 2 + (i / n_attacks) * 6  # De 2 à 8
        mixed.append(np.random.randn(encoding_dim) * intensity)
    scenarios['Progressive'] = np.array(mixed)
    
    return scenarios


def test_threshold_robustness():
    """
    Tester la robustesse de différents thresholds contre le poisoning
    """
    
    print("=" * 70)
    print("TEST 1: ROBUSTESSE DES THRESHOLDS CONTRE LE POISONING")
    print("=" * 70)
    
    # Paramètres
    memory_size = 50
    encoding_dim = 5
    k_neighbors = 3
    gamma = 0.5
    
    # Générer mémoire normale
    np.random.seed(42)
    normal_encodings = np.random.randn(memory_size, encoding_dim)
    
    # Thresholds à tester
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    
    # Scénarios d'attaque
    scenarios = generate_attack_scenarios(encoding_dim)
    
    # Résultats
    results = {scenario: [] for scenario in scenarios}
    
    print(f"\n📊 Configuration:")
    print(f"   Memory size: {memory_size}")
    print(f"   Encoding dim: {encoding_dim}")
    print(f"   K neighbors: {k_neighbors}")
    print(f"   Gamma: {gamma}")
    print(f"   Nombre de scénarios: {len(scenarios)}")
    
    # Tester chaque combinaison threshold × scenario
    for scenario_name, anomaly_stream in scenarios.items():
        print(f"\n{'─'*70}")
        print(f"Scénario: {scenario_name}")
        print(f"{'─'*70}")
        
        for threshold in thresholds:
            # Créer et initialiser la mémoire
            memory = MemoryModule(
                memory_size=memory_size,
                encoding_dim=encoding_dim,
                k_neighbors=k_neighbors,
                gamma=gamma
            )
            memory.initialize(normal_encodings)
            
            # Simuler l'attaque
            result = memory.simulate_poisoning(
                anomaly_stream, 
                threshold, 
                verbose=False
            )
            
            results[scenario_name].append({
                'threshold': threshold,
                'protection_rate': result['protection_rate'],
                'contamination_rate': result['contamination_rate'],
                'avg_score': result['scores'].mean()
            })
            
            print(f"   β={threshold:>5.1f} → Protection: {result['protection_rate']:>5.1f}%, "
                  f"Contamination: {result['contamination_rate']:>5.1f}%")
    
    # Visualisation
    print(f"\n📈 Génération des visualisations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Analyse de Résistance au Memory Poisoning', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    # Subplot 1: Protection rate par scénario
    ax = axes[0, 0]
    for scenario_name in scenarios:
        data = results[scenario_name]
        thresholds_list = [d['threshold'] for d in data]
        protection_rates = [d['protection_rate'] for d in data]
        ax.plot(thresholds_list, protection_rates, marker='o', label=scenario_name, linewidth=2)
    
    ax.set_xlabel('Threshold (β)', fontsize=11)
    ax.set_ylabel('Taux de Protection (%)', fontsize=11)
    ax.set_title('Protection vs Threshold', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='80% cible')
    
    # Subplot 2: Contamination rate par scénario
    ax = axes[0, 1]
    for scenario_name in scenarios:
        data = results[scenario_name]
        thresholds_list = [d['threshold'] for d in data]
        contamination_rates = [d['contamination_rate'] for d in data]
        ax.plot(thresholds_list, contamination_rates, marker='s', label=scenario_name, linewidth=2)
    
    ax.set_xlabel('Threshold (β)', fontsize=11)
    ax.set_ylabel('Taux de Contamination (%)', fontsize=11)
    ax.set_title('Contamination vs Threshold', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='20% limite')
    
    # Subplot 3: Heatmap protection (Threshold × Scenario)
    ax = axes[0, 2]
    protection_matrix = []
    for scenario_name in scenarios:
        data = results[scenario_name]
        protection_matrix.append([d['protection_rate'] for d in data])
    
    im = ax.imshow(protection_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'{t:.1f}' for t in thresholds], rotation=45)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios.keys())
    ax.set_xlabel('Threshold (β)', fontsize=11)
    ax.set_title('Heatmap: Taux de Protection (%)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Subplot 4: Comparaison des scores moyens
    ax = axes[1, 0]
    x_pos = np.arange(len(scenarios))
    avg_scores = []
    for scenario_name in scenarios:
        # Prendre le score moyen pour threshold=2.0 (valeur typique)
        data = [d for d in results[scenario_name] if d['threshold'] == 2.0][0]
        avg_scores.append(data['avg_score'])
    
    bars = ax.bar(x_pos, avg_scores, alpha=0.7, color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios.keys(), rotation=45, ha='right')
    ax.set_ylabel('Score Moyen', fontsize=11)
    ax.set_title('Score d\'Anomalie Moyen par Scénario (β=2.0)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, avg_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 5: Trade-off Protection vs Contamination
    ax = axes[1, 1]
    for scenario_name in scenarios:
        data = results[scenario_name]
        protection_rates = [d['protection_rate'] for d in data]
        contamination_rates = [d['contamination_rate'] for d in data]
        ax.scatter(contamination_rates, protection_rates, s=100, alpha=0.6, label=scenario_name)
    
    ax.set_xlabel('Contamination (%)', fontsize=11)
    ax.set_ylabel('Protection (%)', fontsize=11)
    ax.set_title('Trade-off Protection vs Contamination', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axvline(x=20, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=80, color='g', linestyle='--', alpha=0.5)
    
    # Zone optimale
    ax.fill_between([0, 20], 80, 100, alpha=0.2, color='green', label='Zone optimale')
    
    # Subplot 6: Recommandations (texte)
    ax = axes[1, 2]
    ax.axis('off')
    
    # Trouver le meilleur threshold global
    best_threshold = None
    best_avg_protection = 0
    
    for threshold in thresholds:
        avg_protection = np.mean([
            [d for d in results[s] if d['threshold'] == threshold][0]['protection_rate']
            for s in scenarios
        ])
        if avg_protection > best_avg_protection:
            best_avg_protection = avg_protection
            best_threshold = threshold
    
    recommendations = f"""
    RECOMMANDATIONS
    {'─'*30}
    
    Meilleur Threshold:
      β = {best_threshold}
    
    Protection moyenne:
      {best_avg_protection:.1f}%
    
    Scénario le plus difficile:
      {max(scenarios.keys(), 
           key=lambda s: results[s][0]['contamination_rate'])}
    
    Scénario le plus facile:
      {min(scenarios.keys(), 
           key=lambda s: results[s][0]['contamination_rate'])}
    
    {'─'*30}
    Pour une protection ≥80%:
      Utiliser β ≥ {min([t for t in thresholds if all(
        [d for d in results[s] if d['threshold'] == t][0]['protection_rate'] >= 80
        for s in scenarios
      )], default='Non atteint')}
    """
    
    ax.text(0.1, 0.5, recommendations, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/memory_poisoning_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Figure sauvegardée: results/memory_poisoning_analysis.png")
    
    return results


def test_gamma_impact_on_poisoning():
    """
    Tester l'impact de gamma sur la résistance au poisoning
    """
    
    print("\n" + "=" * 70)
    print("TEST 2: IMPACT DE GAMMA SUR LA RÉSISTANCE AU POISONING")
    print("=" * 70)
    
    # Paramètres
    memory_size = 50
    encoding_dim = 5
    k_neighbors = 3
    threshold = 2.0
    
    # Valeurs de gamma
    gamma_values = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    # Générer données
    np.random.seed(42)
    normal_encodings = np.random.randn(memory_size, encoding_dim)
    anomaly_stream = np.random.randn(100, encoding_dim) * 4 + 6
    
    results = []
    
    print(f"\n📊 Configuration:")
    print(f"   Threshold fixe: β = {threshold}")
    print(f"   Nombre d'attaques: {len(anomaly_stream)}")
    
    for gamma in gamma_values:
        print(f"\n   Testing γ = {gamma}...", end=' ')
        
        memory = MemoryModule(
            memory_size=memory_size,
            encoding_dim=encoding_dim,
            k_neighbors=k_neighbors,
            gamma=gamma
        )
        memory.initialize(normal_encodings)
        
        result = memory.simulate_poisoning(anomaly_stream, threshold, verbose=False)
        
        results.append({
            'gamma': gamma,
            'protection_rate': result['protection_rate'],
            'contamination_rate': result['contamination_rate']
        })
        
        print(f"Protection: {result['protection_rate']:.1f}%")
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    gammas = [r['gamma'] for r in results]
    protections = [r['protection_rate'] for r in results]
    contaminations = [r['contamination_rate'] for r in results]
    
    # Protection
    ax1.plot(gammas, protections, 'o-', linewidth=2, markersize=10, color='green')
    ax1.set_xlabel('Gamma (γ)', fontsize=12)
    ax1.set_ylabel('Taux de Protection (%)', fontsize=12)
    ax1.set_title('Impact de γ sur la Protection', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.axhline(y=80, color='g', linestyle='--', alpha=0.5)
    
    # Contamination
    ax2.plot(gammas, contaminations, 's-', linewidth=2, markersize=10, color='red')
    ax2.set_xlabel('Gamma (γ)', fontsize=12)
    ax2.set_ylabel('Taux de Contamination (%)', fontsize=12)
    ax2.set_title('Impact de γ sur la Contamination', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=20, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/gamma_poisoning_impact.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure sauvegardée: results/gamma_poisoning_impact.png")
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    print("\n🚀 DÉMARRAGE DES TESTS DE MEMORY POISONING")
    print("=" * 70)
    
    # Test 1: Robustesse des thresholds
    results_thresholds = test_threshold_robustness()
    
    # Test 2: Impact de gamma
    results_gamma = test_gamma_impact_on_poisoning()
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS TERMINÉS")
    print("=" * 70)
    print("\n📁 Fichiers générés:")
    print("   - results/memory_poisoning_analysis.png")
    print("   - results/gamma_poisoning_impact.png")
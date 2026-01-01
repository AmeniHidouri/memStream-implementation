"""
Test de l'impact du coefficient gamma sur la détection d'anomalies
"""

import numpy as np
import matplotlib.pyplot as plt
from src.memory import MemoryModule

def test_gamma_impact():
    """Tester différentes valeurs de gamma"""
    
    print("=" * 60)
    print("TEST IMPACT DU COEFFICIENT GAMMA")
    print("=" * 60)
    
    # Paramètres
    memory_size = 50
    encoding_dim = 5
    k_neighbors = 3
    gamma_values = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    # Générer des encodings normaux (distribution normale centrée)
    np.random.seed(42)
    normal_encodings = np.random.randn(memory_size, encoding_dim)
    
    # Encodings de test
    normal_test = np.random.randn(encoding_dim)  # Normal
    anomaly_mild = np.random.randn(encoding_dim) * 2 + 2  # Anomalie légère
    anomaly_strong = np.random.randn(encoding_dim) * 4 + 5  # Anomalie forte
    
    results = {
        'gamma': [],
        'normal_score': [],
        'mild_anomaly_score': [],
        'strong_anomaly_score': []
    }
    
    # Tester chaque valeur de gamma
    for gamma in gamma_values:
        print(f"\n{'=' * 40}")
        print(f"Gamma = {gamma}")
        print(f"{'=' * 40}")
        
        # Créer le module mémoire
        memory = MemoryModule(
            memory_size=memory_size,
            encoding_dim=encoding_dim,
            k_neighbors=k_neighbors,
            gamma=gamma
        )
        
        # Initialiser
        memory.initialize(normal_encodings)
        
        # Calculer les scores
        score_normal = memory.compute_score(normal_test)
        score_mild = memory.compute_score(anomaly_mild)
        score_strong = memory.compute_score(anomaly_strong)
        
        print(f"  Normal:          {score_normal:.4f}")
        print(f"  Anomalie légère: {score_mild:.4f} (ratio: {score_mild/score_normal:.2f}x)")
        print(f"  Anomalie forte:  {score_strong:.4f} (ratio: {score_strong/score_normal:.2f}x)")
        
        # Sauvegarder
        results['gamma'].append(gamma)
        results['normal_score'].append(score_normal)
        results['mild_anomaly_score'].append(score_mild)
        results['strong_anomaly_score'].append(score_strong)
    
    # Visualisation
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Scores absolus
    plt.subplot(1, 2, 1)
    plt.plot(results['gamma'], results['normal_score'], 'o-', label='Normal', linewidth=2)
    plt.plot(results['gamma'], results['mild_anomaly_score'], 's-', label='Anomalie légère', linewidth=2)
    plt.plot(results['gamma'], results['strong_anomaly_score'], '^-', label='Anomalie forte', linewidth=2)
    plt.xlabel('Gamma (γ)', fontsize=12)
    plt.ylabel('Score d\'anomalie', fontsize=12)
    plt.title('Impact de γ sur les scores d\'anomalie', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 2: Ratios
    plt.subplot(1, 2, 2)
    ratios_mild = np.array(results['mild_anomaly_score']) / np.array(results['normal_score'])
    ratios_strong = np.array(results['strong_anomaly_score']) / np.array(results['normal_score'])
    
    plt.plot(results['gamma'], ratios_mild, 's-', label='Ratio (légère/normal)', linewidth=2)
    plt.plot(results['gamma'], ratios_strong, '^-', label='Ratio (forte/normal)', linewidth=2)
    plt.xlabel('Gamma (γ)', fontsize=12)
    plt.ylabel('Ratio de séparation', fontsize=12)
    plt.title('Capacité de discrimination', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Seuil 2x')
    
    plt.tight_layout()
    plt.savefig('results/gamma_impact.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure sauvegardée: results/gamma_impact.png")
    
    # Conclusions
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    
    best_gamma_idx = np.argmax(ratios_strong)
    best_gamma = results['gamma'][best_gamma_idx]
    
    print(f"Meilleure valeur de γ pour la séparation: {best_gamma}")
    print(f"Ratio de séparation maximal: {ratios_strong[best_gamma_idx]:.2f}x")
    
    if best_gamma > 0.5:
        print("\n💡 RECOMMANDATION: Utiliser γ élevé (0.7-0.9)")
        print("   → Les voisins proches ont plus de poids")
    else:
        print("\n💡 RECOMMANDATION: Utiliser γ faible (0.0-0.3)")
        print("   → Tous les voisins ont un poids similaire")
    
    return results


def test_memory_poisoning_resistance():
    """
    Tester la résistance à l'empoisonnement de la mémoire
    Tâche pour Personne 2
    """
    
    print("\n" + "=" * 60)
    print("TEST RÉSISTANCE AU MEMORY POISONING")
    print("=" * 60)
    
    # Paramètres
    memory_size = 50
    encoding_dim = 5
    k_neighbors = 3
    gamma = 0.5
    
    # Créer mémoire
    memory = MemoryModule(
        memory_size=memory_size,
        encoding_dim=encoding_dim,
        k_neighbors=k_neighbors,
        gamma=gamma
    )
    
    # Initialiser avec données normales
    np.random.seed(42)
    normal_encodings = np.random.randn(memory_size, encoding_dim)
    memory.initialize(normal_encodings)
    
    print(f"\n1. Mémoire initialisée avec {memory_size} encodings normaux")
    
    # Scénario de poisoning: tenter d'injecter des anomalies
    print(f"\n2. Tentative d'injection de 20 anomalies dans le stream")
    
    thresholds = [0.5, 1.0, 2.0, 5.0]
    results = []
    
    for threshold in thresholds:
        print(f"\n   Threshold β = {threshold}")
        
        # Réinitialiser la mémoire
        memory.initialize(normal_encodings)
        
        accepted = 0
        rejected = 0
        
        # Générer 20 anomalies
        for i in range(20):
            anomaly = np.random.randn(encoding_dim) * 3 + 4
            score = memory.compute_score(anomaly)
            updated = memory.update_fifo(anomaly, score, threshold)
            
            if updated:
                accepted += 1
            else:
                rejected += 1
        
        print(f"      ✅ Acceptées: {accepted}")
        print(f"      ❌ Rejetées: {rejected}")
        print(f"      Protection: {(rejected/20)*100:.1f}%")
        
        results.append({
            'threshold': threshold,
            'accepted': accepted,
            'rejected': rejected,
            'protection': (rejected/20)*100
        })
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    thresholds_list = [r['threshold'] for r in results]
    protection = [r['protection'] for r in results]
    
    plt.bar(range(len(thresholds_list)), protection, color='steelblue', alpha=0.7)
    plt.xlabel('Threshold (β)', fontsize=12)
    plt.ylabel('Taux de protection (%)', fontsize=12)
    plt.title('Résistance au Memory Poisoning', fontsize=14, fontweight='bold')
    plt.xticks(range(len(thresholds_list)), thresholds_list)
    plt.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='80% protection')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/memory_poisoning.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure sauvegardée: results/memory_poisoning.png")
    
    # Recommandations
    print("\n" + "=" * 60)
    print("RECOMMANDATIONS")
    print("=" * 60)
    
    best_threshold = max(results, key=lambda x: x['protection'])
    print(f"Meilleur threshold pour protection: β = {best_threshold['threshold']}")
    print(f"Protection: {best_threshold['protection']:.1f}%")
    
    if best_threshold['protection'] >= 80:
        print("\n✅ Bonne résistance au poisoning!")
    else:
        print("\n⚠️ Augmenter le threshold pour plus de protection")
    
    return results


if __name__ == "__main__":
    # Créer dossier results si nécessaire
    import os
    os.makedirs('results', exist_ok=True)
    
    # Test 1: Impact de gamma
    print("TEST 1: IMPACT DU GAMMA")
    gamma_results = test_gamma_impact()
    
    # Test 2: Memory poisoning
    print("\n\n" + "=" * 60)
    print("TEST 2: MEMORY POISONING")
    print("=" * 60)
    poison_results = test_memory_poisoning_resistance()
    
    print("\n\n" + "=" * 60)
    print("✅ TOUS LES TESTS TERMINÉS")
    print("=" * 60)
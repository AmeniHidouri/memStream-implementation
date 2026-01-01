"""
Ablation Study: Comparaison des politiques de remplacement mémoire
FIFO vs LRU vs Random
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.memory import MemoryModule
import time

class MemoryModuleLRU(MemoryModule):
    """Extension avec politique LRU (Least Recently Used)"""
    
    def __init__(self, memory_size, encoding_dim, k_neighbors=3, gamma=0.0):
        super().__init__(memory_size, encoding_dim, k_neighbors, gamma)
        self.access_times = np.zeros(memory_size)
        self.current_time = 0
    
    def update_lru(self, encoding, score, threshold):
        """Mettre à jour avec politique LRU"""
        if score >= threshold:
            return False
        
        self.current_time += 1
        
        if self.current_size < self.memory_size:
            # Mémoire pas pleine
            self.memory[self.current_size] = encoding
            self.access_times[self.current_size] = self.current_time
            self.current_size += 1
        else:
            # LRU: remplacer l'élément le moins récemment utilisé
            lru_idx = np.argmin(self.access_times[:self.current_size])
            self.memory[lru_idx] = encoding
            self.access_times[lru_idx] = self.current_time
        
        # Refitter KNN
        if self.current_size > 0:
            self.knn.fit(self.memory[:self.current_size])
            self.is_fitted = True
        
        return True
    
    def compute_score(self, encoding):
        """Override pour mettre à jour access_times"""
        if not self.is_fitted or self.current_size == 0:
            return 0.0
        
        # Mettre à jour le temps d'accès des K voisins
        encoding_reshaped = encoding.reshape(1, -1)
        distances, indices = self.knn.kneighbors(encoding_reshaped)
        
        self.current_time += 1
        for idx in indices[0]:
            if idx < self.current_size:
                self.access_times[idx] = self.current_time
        
        # Calculer score normalement
        return super().compute_score(encoding)


class MemoryModuleRandom(MemoryModule):
    """Extension avec politique Random Replacement"""
    
    def update_random(self, encoding, score, threshold):
        """Mettre à jour avec politique Random"""
        if score >= threshold:
            return False
        
        if self.current_size < self.memory_size:
            # Mémoire pas pleine
            self.memory[self.current_size] = encoding
            self.current_size += 1
        else:
            # Random: remplacer un élément aléatoire
            random_idx = np.random.randint(0, self.memory_size)
            self.memory[random_idx] = encoding
        
        # Refitter KNN
        if self.current_size > 0:
            self.knn.fit(self.memory[:self.current_size])
            self.is_fitted = True
        
        return True
    
class MemoryModuleWeightedFIFO(MemoryModule):
    """Extension : Les éléments plus récents ont plus de poids dans le KNN"""
    
    def compute_score(self, encoding):
        if not self.is_fitted or self.current_size == 0:
            return 0.0
        
        encoding_reshaped = encoding.reshape(1, -1)
        distances, indices = self.knn.kneighbors(encoding_reshaped)
        distances = distances[0]
        indices = indices[0]

        # Calcul des poids temporels (plus l'indice est proche du pointer, plus il est récent)
        # On peut utiliser une décroissance linéaire ou exponentielle
        time_weights = []
        for idx in indices:
            # Distance temporelle par rapport au pointeur actuel
            age = (self.pointer - 1 - idx) % self.memory_size
            decay_factor = 0.01  # Stable
            #decay_factor = 0.05  # Équilibré
            #decay_factor = 0.2   # Réactif
            time_weights.append(np.exp(- decay_factor * age)) # Décroissance exponentielle de l'importance
        
        time_weights = np.array(time_weights)
        time_weights /= time_weights.sum() # Normalisation

        # Score final combinant distance L1 et poids temporel
        return float(np.sum(distances * time_weights))


def generate_stream_with_drift(n_samples=1000, encoding_dim=5, drift_point=500):
    """
    Générer un stream avec concept drift
    
    Args:
        n_samples: nombre total d'échantillons
        encoding_dim: dimension
        drift_point: point où le drift commence
    
    Returns:
        stream: (n_samples, encoding_dim)
        labels: (n_samples,) - 0=normal, 1=anomalie
    """
    np.random.seed(42)
    
    stream = []
    labels = []
    
    # Phase 1: Distribution normale centrée en 0
    for i in range(drift_point):
        if np.random.rand() < 0.1:  # 10% anomalies
            x = np.random.randn(encoding_dim) * 3 + 5  # Anomalie
            labels.append(1)
        else:
            x = np.random.randn(encoding_dim)  # Normal
            labels.append(0)
        stream.append(x)
    
    # Phase 2: Drift - distribution se déplace vers (3, 3, ...)
    drift_target = np.ones(encoding_dim) * 3
    for i in range(drift_point, n_samples):
        progress = (i - drift_point) / (n_samples - drift_point)
        drift_amount = drift_target * progress
        
        if np.random.rand() < 0.1:  # 10% anomalies
            x = np.random.randn(encoding_dim) * 3 + 8  # Anomalie
            labels.append(1)
        else:
            x = np.random.randn(encoding_dim) + drift_amount  # Normal avec drift
            labels.append(0)
        stream.append(x)
    
    return np.array(stream), np.array(labels)


def evaluate_policy(policy_name, memory_class, update_method, 
                stream, labels, memory_size=50, k_neighbors=3, gamma=0.5):
    """
    Évaluer une politique de remplacement
    
    Returns:
        scores: array de scores d'anomalie
        auc: AUC-ROC
        runtime: temps d'exécution
    """
    
    print(f"\n{'='*50}")
    print(f"Politique: {policy_name}")
    print(f"{'='*50}")
    
    # Créer mémoire
    encoding_dim = stream.shape[1]
    memory = memory_class(
        memory_size=memory_size,
        encoding_dim=encoding_dim,
        k_neighbors=k_neighbors,
        gamma=gamma
    )
    
    # Initialiser avec les 50 premiers échantillons normaux
    init_encodings = stream[:memory_size]
    memory.initialize(init_encodings)
    
    # Stream processing
    scores = []
    start_time = time.time()
    threshold = 0.5
    
    for i, encoding in enumerate(stream):
        # Calculer score
        score = memory.compute_score(encoding)
        scores.append(score)
        
        # Mettre à jour mémoire selon politique
        if policy_name == ["FIFO", "Weighted-FIFO"]:
            memory.update_fifo(encoding, score, threshold)
        elif policy_name == "LRU":
            memory.update_lru(encoding, score, threshold)
        elif policy_name == "Random":
            memory.update_random(encoding, score, threshold)
    
    runtime = time.time() - start_time
    
    # Calculer AUC
    scores = np.array(scores)
    auc = roc_auc_score(labels, scores)
    
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Runtime: {runtime:.3f}s")
    print(f"  Score moyen (normal): {scores[labels==0].mean():.4f}")
    print(f"  Score moyen (anomalie): {scores[labels==1].mean():.4f}")
    
    return scores, auc, runtime


def compare_policies():
    """Comparer les 3 politiques de remplacement"""
    
    print("="*60)
    print("ABLATION STUDY: POLITIQUES DE REMPLACEMENT MÉMOIRE")
    print("="*60)
    
    # Générer stream avec drift
    print("\n1. Génération du stream avec concept drift")
    n_samples = 10000
    encoding_dim = 5
    drift_point = 5000
    
    stream, labels = generate_stream_with_drift(
        n_samples=n_samples,
        encoding_dim=encoding_dim,
        drift_point=drift_point
    )
    
    print(f"   Total échantillons: {n_samples}")
    print(f"   Point de drift: {drift_point}")
    print(f"   Anomalies: {labels.sum()} ({(labels.sum()/len(labels)*100):.1f}%)")
    
    # Paramètres
    memory_size = 100
    k_neighbors = 3
    gamma = 0.5
    
    print(f"\n2. Paramètres")
    print(f"   Memory size: {memory_size}")
    print(f"   K neighbors: {k_neighbors}")
    print(f"   Gamma: {gamma}")
    
    # Évaluer chaque politique
    print(f"\n3. Évaluation des politiques")
    
    results = {}
    
    # FIFO
    scores_fifo, auc_fifo, time_fifo = evaluate_policy(
        "FIFO", MemoryModule, "update_fifo",
        stream, labels, memory_size, k_neighbors, gamma
    )
    results['FIFO'] = {'scores': scores_fifo, 'auc': auc_fifo, 'time': time_fifo}
    
    # Weighted-FIFO
    scores_wfifo, auc_wfifo, time_wfifo = evaluate_policy(
        "Weighted-FIFO", MemoryModuleWeightedFIFO, "update_fifo",
        stream, labels, memory_size, k_neighbors, gamma
    )
    results['Weighted-FIFO'] = {'scores': scores_wfifo, 'auc': auc_wfifo, 'time': time_wfifo}
    
    # LRU
    scores_lru, auc_lru, time_lru = evaluate_policy(
        "LRU", MemoryModuleLRU, "update_lru",
        stream, labels, memory_size, k_neighbors, gamma
    )
    results['LRU'] = {'scores': scores_lru, 'auc': auc_lru, 'time': time_lru}
    
    # Random
    scores_random, auc_random, time_random = evaluate_policy(
        "Random", MemoryModuleRandom, "update_random",
        stream, labels, memory_size, k_neighbors, gamma
    )
    results['Random'] = {'scores': scores_random, 'auc': auc_random, 'time': time_random}
    
    # Visualisation
    print(f"\n4. Génération des visualisations")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Scores au fil du temps
    ax = axes[0, 0]
    window = 50
    for policy in ['FIFO', 'Weighted-FIFO', 'LRU', 'Random']:
        scores = results[policy]['scores']
        # Moyenne mobile
        smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=policy, linewidth=2, alpha=0.8)
    
    ax.axvline(x=drift_point, color='r', linestyle='--', alpha=0.5, label='Drift')
    ax.set_xlabel('Temps', fontsize=12)
    ax.set_ylabel('Score d\'anomalie (moyenne mobile)', fontsize=12)
    ax.set_title('Évolution des scores au fil du temps', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Subplot 2: Comparaison AUC
    ax = axes[0, 1]
    policies = ['FIFO', 'Weighted-FIFO', 'LRU', 'Random']
    aucs = [results[p]['auc'] for p in policies]
    colors = ['steelblue', 'darkorange', 'green']
    
    bars = ax.bar(policies, aucs, color=colors, alpha=0.7)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Comparaison des performances', fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Annoter les barres
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Subplot 3: Runtime comparison
    ax = axes[1, 0]
    times = [results[p]['time'] for p in policies]
    bars = ax.bar(policies, times, color=colors, alpha=0.7)
    ax.set_ylabel('Temps d\'exécution (s)', fontsize=12)
    ax.set_title('Comparaison du temps d\'exécution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.2f}s', ha='center', va='bottom', fontsize=11)
    
    # Subplot 4: Tableau récapitulatif
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [
        ['Politique', 'AUC-ROC', 'Temps (s)', 'Gagnant?'],
        ['FIFO', f"{results['FIFO']['auc']:.4f}", f"{results['FIFO']['time']:.2f}", ''],
        ['Weighted-FIFO', f"{results['Weighted-FIFO']['auc']:.4f}", f"{results['Weighted-FIFO']['time']:.2f}", ''],
        ['LRU', f"{results['LRU']['auc']:.4f}", f"{results['LRU']['time']:.2f}", ''],
        ['Random', f"{results['Random']['auc']:.4f}", f"{results['Random']['time']:.2f}", '']
    ]
    
    # Identifier le gagnant
    best_policy = max(results.items(), key=lambda x: x[1]['auc'])[0]
    for i, row in enumerate(table_data[1:], 1):
        if row[0] == best_policy:
            row[3] = '✅'
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Résumé des résultats', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/ablation_replacement_policies.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Figure sauvegardée: results/ablation_replacement_policies.png")
    
    # Conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    print(f"\n🏆 Meilleure politique: {best_policy}")
    print(f"   AUC-ROC: {results[best_policy]['auc']:.4f}")
    
    improvement = (results[best_policy]['auc'] - min(aucs)) / min(aucs) * 100
    print(f"   Amélioration vs pire: {improvement:.1f}%")
    
    if best_policy == "FIFO":
        print("\n✅ Résultat conforme au paper: FIFO est optimal")
        print("   Raison: Adaptation naturelle au concept drift")
    else:
        print(f"\n⚠️ Résultat inattendu: {best_policy} surpasse FIFO")
        print("   Vérifier les paramètres et l'implémentation")
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    results = compare_policies()
    
    print("\n" + "="*60)
    print("✅ ABLATION STUDY TERMINÉE")
    print("="*60)
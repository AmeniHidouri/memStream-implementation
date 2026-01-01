"""
Memory Module pour MemStream avec FIFO + Memory Poisoning Simulation
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

class MemoryModule:
    """
    Memory Module avec K-Nearest Neighbors et FIFO update
    """
    def __init__(self, memory_size, encoding_dim, k_neighbors=3, gamma=0.0):
        """
        Args:
            memory_size: taille de la mémoire (N dans le paper)
            encoding_dim: dimension des encodings
            k_neighbors: nombre de voisins (K dans le paper)
            gamma: coefficient de discount (γ dans le paper)
        """
        self.memory_size = memory_size
        self.encoding_dim = encoding_dim
        self.k = k_neighbors
        self.gamma = gamma
        
        # Initialiser la mémoire
        self.memory = np.zeros((memory_size, encoding_dim))
        self.current_size = 0
        self.pointer = 0  # Pointeur FIFO
        
        # Statistiques pour poisoning
        self.rejected_count = 0
        self.accepted_count = 0
        
        # KNN avec distance L1 (Manhattan)
        self.knn = NearestNeighbors(n_neighbors=k_neighbors, metric='l1')
        self.is_fitted = False
    
    def initialize(self, initial_encodings):
        """
        Initialiser la mémoire avec des encodings initiaux
        
        Args:
            initial_encodings: numpy array (memory_size, encoding_dim)
        """
        assert len(initial_encodings) >= self.memory_size, \
            f"Besoin d'au moins {self.memory_size} encodings"
        
        # Prendre les premiers memory_size encodings
        self.memory = np.array(initial_encodings[:self.memory_size])
        self.current_size = self.memory_size
        self.pointer = 0
        
        # Réinitialiser les stats
        self.rejected_count = 0
        self.accepted_count = 0
        
        # Fitter le KNN
        self.knn.fit(self.memory)
        self.is_fitted = True
        
        print(f"✅ Mémoire initialisée avec {self.memory_size} encodings")
    
    def compute_score(self, encoding):
        """
        Calculer le score d'anomalie (discounted distance)
        
        Args:
            encoding: numpy array (encoding_dim,)
        
        Returns:
            score: float (plus élevé = plus anormal)
        """
        if not self.is_fitted or self.current_size == 0:
            return 0.0
        
        # Reshape pour sklearn
        encoding = np.array(encoding).reshape(1, -1)
        
        # Trouver les K plus proches voisins
        distances, indices = self.knn.kneighbors(encoding)
        distances = distances[0]  # Shape: (k,)
        
        # Calculer le discounted score
        if self.gamma == 0:
            # Pas de discount: moyenne simple
            score = np.mean(distances)
        else:
            # Avec discount: moyenne pondérée
            weights = np.array([self.gamma ** i for i in range(self.k)])
            weights = weights / np.sum(weights)  # Normaliser
            score = np.sum(distances * weights)
        
        return float(score)
    
    def update_fifo(self, encoding, score, threshold):
        """
        Mettre à jour la mémoire avec politique FIFO
        
        Args:
            encoding: numpy array (encoding_dim,)
            score: float, score d'anomalie
            threshold: float (β dans le paper)
        
        Returns:
            updated: bool, True si la mémoire a été mise à jour
        """
        # Ne mettre à jour que si score < threshold
        if score >= threshold:
            self.rejected_count += 1
            return False
        
        self.accepted_count += 1
        
        if self.current_size < self.memory_size:
            # Mémoire pas encore pleine
            self.memory[self.current_size] = encoding
            self.current_size += 1
        else:
            # FIFO: remplacer l'élément le plus ancien
            self.memory[self.pointer] = encoding
            self.pointer = (self.pointer + 1) % self.memory_size
        
        # Refitter le KNN avec la mémoire actuelle
        if self.current_size > 0:
            self.knn.fit(self.memory[:self.current_size])
            self.is_fitted = True
        
        return True
    
    def simulate_poisoning(self, anomaly_stream, threshold, verbose=True):
        """
        Simuler une attaque de memory poisoning
        
        Args:
            anomaly_stream: numpy array (n_anomalies, encoding_dim) - encodings anormaux
            threshold: float (β) - seuil de protection
            verbose: afficher les détails
        
        Returns:
            results: dict avec statistiques de l'attaque
        """
        if verbose:
            print("\n" + "="*50)
            print("SIMULATION MEMORY POISONING ATTACK")
            print("="*50)
            print(f"Nombre d'anomalies injectées: {len(anomaly_stream)}")
            print(f"Threshold de protection (β): {threshold}")
        
        # Sauvegarder l'état initial
        initial_memory = self.memory[:self.current_size].copy()
        initial_rejected = self.rejected_count
        initial_accepted = self.accepted_count
        
        # Tenter d'injecter chaque anomalie
        scores = []
        for i, anomaly in enumerate(anomaly_stream):
            score = self.compute_score(anomaly)
            scores.append(score)
            self.update_fifo(anomaly, score, threshold)
        
        # Calculer les statistiques
        newly_accepted = self.accepted_count - initial_accepted
        newly_rejected = self.rejected_count - initial_rejected
        
        # Calculer la contamination de la mémoire
        final_memory = self.memory[:self.current_size]
        memory_changed = not np.array_equal(initial_memory, final_memory[:len(initial_memory)])
        
        results = {
            'total_attacks': len(anomaly_stream),
            'accepted': newly_accepted,
            'rejected': newly_rejected,
            'protection_rate': (newly_rejected / len(anomaly_stream)) * 100,
            'contamination_rate': (newly_accepted / self.memory_size) * 100,
            'scores': np.array(scores),
            'memory_changed': memory_changed,
            'threshold': threshold
        }
        
        if verbose:
            print(f"\n📊 Résultats:")
            print(f"   ✅ Acceptées: {newly_accepted} ({(newly_accepted/len(anomaly_stream)*100):.1f}%)")
            print(f"   ❌ Rejetées: {newly_rejected} ({(newly_rejected/len(anomaly_stream)*100):.1f}%)")
            print(f"   🛡️ Taux de protection: {results['protection_rate']:.1f}%")
            print(f"   ☠️ Taux de contamination: {results['contamination_rate']:.1f}%")
            print(f"   Score moyen des attaques: {results['scores'].mean():.4f}")
            print(f"   Mémoire modifiée: {memory_changed}")
        
        return results
    
    def get_memory_state(self):
        """Obtenir l'état actuel de la mémoire"""
        return {
            'size': self.current_size,
            'memory': self.memory[:self.current_size].copy(),
            'pointer': self.pointer,
            'accepted': self.accepted_count,
            'rejected': self.rejected_count
        }
    
    def get_statistics(self):
        """Statistiques sur la mémoire"""
        if self.current_size == 0:
            return {}
        
        mem = self.memory[:self.current_size]
        return {
            'size': self.current_size,
            'mean': np.mean(mem, axis=0),
            'std': np.std(mem, axis=0),
            'min': np.min(mem, axis=0),
            'max': np.max(mem, axis=0),
            'accepted_total': self.accepted_count,
            'rejected_total': self.rejected_count
        }


# ============ TEST UNITAIRE ============
def test_memory_with_poisoning():
    """Test du memory module avec simulation de poisoning"""
    print("=" * 50)
    print("TEST MEMORY MODULE + POISONING")
    print("=" * 50)
    
    # Paramètres
    memory_size = 20
    encoding_dim = 5
    k_neighbors = 3
    
    print(f"\n1. Création du module mémoire")
    print(f"   Memory size: {memory_size}")
    print(f"   Encoding dim: {encoding_dim}")
    print(f"   K neighbors: {k_neighbors}")
    
    memory = MemoryModule(
        memory_size=memory_size,
        encoding_dim=encoding_dim,
        k_neighbors=k_neighbors,
        gamma=0.5
    )
    
    # Générer des encodings initiaux (distribution normale)
    print(f"\n2. Génération des encodings initiaux")
    np.random.seed(42)
    initial_encodings = np.random.randn(memory_size, encoding_dim)
    print(f"   Shape: {initial_encodings.shape}")
    
    # Initialiser la mémoire
    print(f"\n3. Initialisation de la mémoire")
    memory.initialize(initial_encodings)
    
    # Générer des anomalies pour l'attaque
    print(f"\n4. Génération d'anomalies pour le poisoning")
    n_anomalies = 50
    anomaly_stream = np.random.randn(n_anomalies, encoding_dim) * 4 + 6
    print(f"   Nombre d'anomalies: {n_anomalies}")
    
    # Tester avec différents thresholds
    print(f"\n5. Test de résistance avec différents thresholds")
    thresholds = [0.5, 1.0, 2.0, 5.0]
    
    for threshold in thresholds:
        # Réinitialiser la mémoire
        memory.initialize(initial_encodings)
        
        # Simuler l'attaque
        results = memory.simulate_poisoning(anomaly_stream, threshold, verbose=False)
        
        print(f"\n   Threshold β = {threshold}:")
        print(f"      Protection: {results['protection_rate']:.1f}%")
        print(f"      Contamination: {results['contamination_rate']:.1f}%")
    
    print("\nTest réussi!")
    print("=" * 50)
    
    return memory


if __name__ == "__main__":
    # Lancer le test
    memory = test_memory_with_poisoning()
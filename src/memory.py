"""
Memory Module pour MemStream avec FIFO
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
        encoding = encoding.reshape(1, -1)
        
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
            return False
        
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
    
    def get_memory_state(self):
        """Obtenir l'état actuel de la mémoire"""
        return {
            'size': self.current_size,
            'memory': self.memory[:self.current_size].copy(),
            'pointer': self.pointer
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
            'max': np.max(mem, axis=0)
        }


# ============ TEST UNITAIRE ============
def test_memory():
    """Test simple du memory module"""
    print("=" * 50)
    print("TEST MEMORY MODULE")
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
    initial_encodings = np.random.randn(memory_size, encoding_dim)
    print(f"   Shape: {initial_encodings.shape}")
    
    # Initialiser la mémoire
    print(f"\n3. Initialisation de la mémoire")
    memory.initialize(initial_encodings)
    state = memory.get_memory_state()
    print(f"   Mémoire remplie: {state['size']}/{memory_size}")
    
    # Test scoring sur un échantillon normal
    print(f"\n4. Test scoring - échantillon NORMAL")
    normal_encoding = np.random.randn(encoding_dim) * 0.5  # Proche de la distribution
    score_normal = memory.compute_score(normal_encoding)
    print(f"   Score: {score_normal:.4f}")
    
    # Test scoring sur un échantillon anormal
    print(f"\n5. Test scoring - échantillon ANORMAL")
    anomaly_encoding = np.random.randn(encoding_dim) * 5.0  # Loin de la distribution
    score_anomaly = memory.compute_score(anomaly_encoding)
    print(f"   Score: {score_anomaly:.4f}")
    print(f"   Ratio anomalie/normal: {score_anomaly/score_normal:.2f}x")
    
    # Test FIFO update
    print(f"\n6. Test FIFO update")
    threshold = 2.0
    print(f"   Threshold: {threshold}")
    
    # Essayer d'ajouter l'échantillon normal
    updated_normal = memory.update_fifo(normal_encoding, score_normal, threshold)
    print(f"   Normal ajouté: {updated_normal}")
    
    # Essayer d'ajouter l'échantillon anormal
    updated_anomaly = memory.update_fifo(anomaly_encoding, score_anomaly, threshold)
    print(f"   Anomalie ajoutée: {updated_anomaly}")
    
    # Statistiques finales
    print(f"\n7. Statistiques mémoire")
    stats = memory.get_statistics()
    print(f"   Taille: {stats['size']}")
    print(f"   Mean encoding: {stats['mean'][:3]}... (3 premières dims)")
    print(f"   Std encoding: {stats['std'][:3]}... (3 premières dims)")
    
    print("\nTest réussi!")
    print("=" * 50)
    
    return memory


if __name__ == "__main__":
    # Lancer le test
    memory = test_memory()

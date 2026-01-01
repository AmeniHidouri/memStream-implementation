"""
MemStream: Memory-Based Streaming Anomaly Detection
Intégration complète
"""

import numpy as np
import torch
from src.autoencoder import DenoisingAutoencoder, AETrainer
from src.memory import MemoryModule
from tqdm import tqdm

class MemStream:
    """
    MemStream complet: Autoencoder + Memory Module
    """
    def __init__(self, 
                 input_dim, 
                 encoding_dim, 
                 memory_size,
                 k_neighbors=3,
                 gamma=0.0,
                 threshold=1.0,
                 device='cpu'):
        """
        Args:
            input_dim: dimension des données d'entrée
            encoding_dim: dimension des encodings
            memory_size: taille de la mémoire (N)
            k_neighbors: nombre de voisins (K)
            gamma: coefficient de discount
            threshold: seuil de mise à jour (β)
            device: 'cpu' ou 'cuda'
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.memory_size = memory_size
        self.threshold = threshold
        self.device = device
        
        # Autoencoder
        self.autoencoder = DenoisingAutoencoder(input_dim, encoding_dim)
        self.ae_trainer = AETrainer(self.autoencoder, device=device)
        
        # Memory Module
        self.memory = MemoryModule(
            memory_size=memory_size,
            encoding_dim=encoding_dim,
            k_neighbors=k_neighbors,
            gamma=gamma
        )
        
        # Stats pour normalisation
        self.mean = None
        self.std = None
        self.is_trained = False
    
    def fit(self, X_train, epochs=100, batch_size=32, noise_factor=0.2, verbose=True):
        """
        Entraîner MemStream sur les données initiales
        
        Args:
            X_train: numpy array (n_samples, input_dim)
            epochs: nombre d'époques pour l'autoencoder
            batch_size: taille des batchs
            noise_factor: facteur de bruit pour le denoising
            verbose: afficher progression
        """
        if verbose:
            print("\n" + "=" * 50)
            print("TRAINING MEMSTREAM")
            print("=" * 50)
        
        # 1. Calculer mean/std pour normalisation
        if verbose:
            print(f"\n1. Calcul des statistiques de normalisation")
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0) + 1e-8  # Éviter division par 0
        
        if verbose:
            print(f"   Mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")
            print(f"   Std range: [{self.std.min():.3f}, {self.std.max():.3f}]")
        
        # 2. Normaliser les données
        if verbose:
            print(f"\n2. Normalisation des données")
        X_normalized = (X_train - self.mean) / self.std
        
        # 3. Entraîner l'autoencoder
        if verbose:
            print(f"\n3. Entraînement de l'autoencoder")
            print(f"   Epochs: {epochs}")
            print(f"   Batch size: {batch_size}")
            print(f"   Noise factor: {noise_factor}")
        
        losses = self.ae_trainer.train(
            X_normalized,
            epochs=epochs,
            batch_size=batch_size,
            noise_factor=noise_factor,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n   Loss initiale: {losses[0]:.4f}")
            print(f"   Loss finale: {losses[-1]:.4f}")
            print(f"   Amélioration: {(1 - losses[-1]/losses[0])*100:.1f}%")
        
        # 4. Générer les encodings pour initialiser la mémoire
        if verbose:
            print(f"\n4. Génération des encodings initiaux")
        
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            encodings = self.autoencoder.encode(X_tensor).cpu().numpy()
        
        if verbose:
            print(f"   Encodings shape: {encodings.shape}")
        
        # 5. Initialiser la mémoire
        if verbose:
            print(f"\n5. Initialisation de la mémoire")
            print(f"   Memory size: {self.memory_size}")
        
        # Prendre memory_size échantillons (ou tous si moins)
        n_samples = min(len(encodings), self.memory_size)
        indices = np.random.choice(len(encodings), size=n_samples, replace=False)
        self.memory.initialize(encodings[indices])
        
        self.is_trained = True
        
        if verbose:
            print("\n" + "=" * 50)
            print("✅ TRAINING TERMINÉ")
            print("=" * 50)
    
    def score_sample(self, x):
        """
        Calculer le score d'anomalie pour UN échantillon
        
        Args:
            x: numpy array (input_dim,)
        
        Returns:
            score: float
        """
        assert self.is_trained, "Le modèle doit être entraîné d'abord (fit)"
        
        # 1. Normaliser
        x_normalized = (x - self.mean) / self.std
        
        # 2. Encoder
        self.autoencoder.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_normalized).unsqueeze(0).to(self.device)
            encoding = self.autoencoder.encode(x_tensor).cpu().numpy()[0]
        
        # 3. Calculer score via mémoire
        score = self.memory.compute_score(encoding)
        
        # 4. Mettre à jour la mémoire
        self.memory.update_fifo(encoding, score, self.threshold)
        
        return score
    
    def predict(self, X_test, verbose=True):
        """
        Prédire les scores pour un batch de données (streaming)
        
        Args:
            X_test: numpy array (n_samples, input_dim)
            verbose: afficher progression
        
        Returns:
            scores: numpy array (n_samples,)
        """
        assert self.is_trained, "Le modèle doit être entraîné d'abord (fit)"
        
        scores = []
        iterator = tqdm(X_test, desc="Scoring") if verbose else X_test
        
        for x in iterator:
            score = self.score_sample(x)
            scores.append(score)
        
        return np.array(scores)


# ============ TEST UNITAIRE ============
def test_memstream():
    """Test complet de MemStream"""
    print("\n" + "=" * 60)
    print("TEST MEMSTREAM COMPLET")
    print("=" * 60)
    
    # 1. Générer des données synthétiques
    print(f"\n1. Génération des données")
    np.random.seed(42)
    
    n_train = 200
    n_test = 100
    input_dim = 10
    
    # Données normales (distribution gaussienne)
    X_train = np.random.randn(n_train, input_dim)
    X_test_normal = np.random.randn(n_test, input_dim)
    
    # Données anormales (distribution décalée)
    X_test_anomaly = np.random.randn(n_test, input_dim) * 3 + 5
    
    print(f"   Train: {X_train.shape}")
    print(f"   Test normal: {X_test_normal.shape}")
    print(f"   Test anomaly: {X_test_anomaly.shape}")
    
    # 2. Créer MemStream
    print(f"\n2. Création de MemStream")
    memstream = MemStream(
        input_dim=input_dim,
        encoding_dim=5,
        memory_size=50,
        k_neighbors=3,
        gamma=0.5,
        threshold=2.0
    )
    print(f"   Input dim: {input_dim}")
    print(f"   Encoding dim: 5")
    print(f"   Memory size: 50")
    
    # 3. Entraîner
    print(f"\n3. Entraînement")
    memstream.fit(X_train, epochs=50, verbose=True)
    
    # 4. Tester sur données normales
    print(f"\n4. Test sur données NORMALES")
    scores_normal = memstream.predict(X_test_normal[:20], verbose=False)
    print(f"   Scores shape: {scores_normal.shape}")
    print(f"   Score moyen: {scores_normal.mean():.4f}")
    print(f"   Score std: {scores_normal.std():.4f}")
    print(f"   Score min: {scores_normal.min():.4f}")
    print(f"   Score max: {scores_normal.max():.4f}")
    
    # 5. Tester sur données anormales
    print(f"\n5. Test sur données ANORMALES")
    scores_anomaly = memstream.predict(X_test_anomaly[:20], verbose=False)
    print(f"   Scores shape: {scores_anomaly.shape}")
    print(f"   Score moyen: {scores_anomaly.mean():.4f}")
    print(f"   Score std: {scores_anomaly.std():.4f}")
    print(f"   Score min: {scores_anomaly.min():.4f}")
    print(f"   Score max: {scores_anomaly.max():.4f}")
    
    # 6. Comparaison
    print(f"\n6. Comparaison Normal vs Anomaly")
    ratio = scores_anomaly.mean() / scores_normal.mean()
    print(f"   Ratio (anomaly/normal): {ratio:.2f}x")
    
    if ratio > 2.0:
        print("    Les anomalies ont des scores plus élevés!")
    else:
        print("     Ratio faible, ajuster les hyperparamètres")
    
    print("\n" + "=" * 60)
    print(" TEST COMPLET RÉUSSI")
    print("=" * 60)
    
    return memstream, scores_normal, scores_anomaly


if __name__ == "__main__":
    # Lancer le test
    memstream, scores_normal, scores_anomaly = test_memstream()

"""
MemStream: Memory-Based Streaming Anomaly Detection
Intégration complète
"""

import numpy as np
import torch
from src.autoencoder import DenoisingAutoencoder, AETrainer
from src.memory import MemoryModule
from tqdm import tqdm

class MemStream:
    """
    MemStream complet: Autoencoder + Memory Module
    """
    def __init__(self, 
                 input_dim, 
                 encoding_dim, 
                 memory_size,
                 k_neighbors=3,
                 gamma=0.0,
                 threshold=1.0,
                 device='cpu'):
        """
        Args:
            input_dim: dimension des données d'entrée
            encoding_dim: dimension des encodings
            memory_size: taille de la mémoire (N)
            k_neighbors: nombre de voisins (K)
            gamma: coefficient de discount
            threshold: seuil de mise à jour (β)
            device: 'cpu' ou 'cuda'
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.memory_size = memory_size
        self.threshold = threshold
        self.device = device
        
        # Autoencoder
        self.autoencoder = DenoisingAutoencoder(input_dim, encoding_dim)
        self.ae_trainer = AETrainer(self.autoencoder, device=device)
        
        # Memory Module
        self.memory = MemoryModule(
            memory_size=memory_size,
            encoding_dim=encoding_dim,
            k_neighbors=k_neighbors,
            gamma=gamma
        )
        
        # Stats pour normalisation
        self.mean = None
        self.std = None
        self.is_trained = False
    
    def fit(self, X_train, epochs=100, batch_size=32, noise_factor=0.2, verbose=True):
        """
        Entraîner MemStream sur les données initiales
        
        Args:
            X_train: numpy array (n_samples, input_dim)
            epochs: nombre d'époques pour l'autoencoder
            batch_size: taille des batchs
            noise_factor: facteur de bruit pour le denoising
            verbose: afficher progression
        """
        if verbose:
            print("\n" + "=" * 50)
            print("TRAINING MEMSTREAM")
            print("=" * 50)
        
        # 1. Calculer mean/std pour normalisation
        if verbose:
            print(f"\n1. Calcul des statistiques de normalisation")
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0) + 1e-8  # Éviter division par 0
        
        if verbose:
            print(f"   Mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")
            print(f"   Std range: [{self.std.min():.3f}, {self.std.max():.3f}]")
        
        # 2. Normaliser les données
        if verbose:
            print(f"\n2. Normalisation des données")
        X_normalized = (X_train - self.mean) / self.std
        
        # 3. Entraîner l'autoencoder
        if verbose:
            print(f"\n3. Entraînement de l'autoencoder")
            print(f"   Epochs: {epochs}")
            print(f"   Batch size: {batch_size}")
            print(f"   Noise factor: {noise_factor}")
        
        losses = self.ae_trainer.train(
            X_normalized,
            epochs=epochs,
            batch_size=batch_size,
            noise_factor=noise_factor,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n   Loss initiale: {losses[0]:.4f}")
            print(f"   Loss finale: {losses[-1]:.4f}")
            print(f"   Amélioration: {(1 - losses[-1]/losses[0])*100:.1f}%")
        
        # 4. Générer les encodings pour initialiser la mémoire
        if verbose:
            print(f"\n4. Génération des encodings initiaux")
        
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            encodings = self.autoencoder.encode(X_tensor).cpu().numpy()
        
        if verbose:
            print(f"   Encodings shape: {encodings.shape}")
        
        # 5. Initialiser la mémoire
        if verbose:
            print(f"\n5. Initialisation de la mémoire")
            print(f"   Memory size: {self.memory_size}")
        
        # Prendre memory_size échantillons (ou tous si moins)
        n_samples = min(len(encodings), self.memory_size)
        indices = np.random.choice(len(encodings), size=n_samples, replace=False)
        self.memory.initialize(encodings[indices])
        
        self.is_trained = True
        
        if verbose:
            print("\n" + "=" * 50)
            print("✅ TRAINING TERMINÉ")
            print("=" * 50)
    
    def score_sample(self, x):
        """
        Calculer le score d'anomalie pour UN échantillon
        
        Args:
            x: numpy array (input_dim,)
        
        Returns:
            score: float
        """
        assert self.is_trained, "Le modèle doit être entraîné d'abord (fit)"
        
        # 1. Normaliser
        x_normalized = (x - self.mean) / self.std
        
        # 2. Encoder
        self.autoencoder.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_normalized).unsqueeze(0).to(self.device)
            encoding = self.autoencoder.encode(x_tensor).cpu().numpy()[0]
        
        # 3. Calculer score via mémoire
        score = self.memory.compute_score(encoding)
        
        # 4. Mettre à jour la mémoire
        self.memory.update_fifo(encoding, score, self.threshold)
        
        return score
    
    def predict(self, X_test, verbose=True):
        """
        Prédire les scores pour un batch de données (streaming)
        
        Args:
            X_test: numpy array (n_samples, input_dim)
            verbose: afficher progression
        
        Returns:
            scores: numpy array (n_samples,)
        """
        assert self.is_trained, "Le modèle doit être entraîné d'abord (fit)"
        
        scores = []
        iterator = tqdm(X_test, desc="Scoring") if verbose else X_test
        
        for x in iterator:
            score = self.score_sample(x)
            scores.append(score)
        
        return np.array(scores)


# ============ TEST UNITAIRE ============
def test_memstream():
    """Test complet de MemStream"""
    print("\n" + "=" * 60)
    print("TEST MEMSTREAM COMPLET")
    print("=" * 60)
    
    # 1. Générer des données synthétiques
    print(f"\n1. Génération des données")
    np.random.seed(42)
    
    n_train = 200
    n_test = 100
    input_dim = 10
    
    # Données normales (distribution gaussienne)
    X_train = np.random.randn(n_train, input_dim)
    X_test_normal = np.random.randn(n_test, input_dim)
    
    # Données anormales (distribution décalée)
    X_test_anomaly = np.random.randn(n_test, input_dim) * 3 + 5
    
    print(f"   Train: {X_train.shape}")
    print(f"   Test normal: {X_test_normal.shape}")
    print(f"   Test anomaly: {X_test_anomaly.shape}")
    
    # 2. Créer MemStream
    print(f"\n2. Création de MemStream")
    memstream = MemStream(
        input_dim=input_dim,
        encoding_dim=5,
        memory_size=50,
        k_neighbors=3,
        gamma=0.5,
        threshold=2.0
    )
    print(f"   Input dim: {input_dim}")
    print(f"   Encoding dim: 5")
    print(f"   Memory size: 50")
    
    # 3. Entraîner
    print(f"\n3. Entraînement")
    memstream.fit(X_train, epochs=50, verbose=True)
    
    # 4. Tester sur données normales
    print(f"\n4. Test sur données NORMALES")
    scores_normal = memstream.predict(X_test_normal[:20], verbose=False)
    print(f"   Scores shape: {scores_normal.shape}")
    print(f"   Score moyen: {scores_normal.mean():.4f}")
    print(f"   Score std: {scores_normal.std():.4f}")
    print(f"   Score min: {scores_normal.min():.4f}")
    print(f"   Score max: {scores_normal.max():.4f}")
    
    # 5. Tester sur données anormales
    print(f"\n5. Test sur données ANORMALES")
    scores_anomaly = memstream.predict(X_test_anomaly[:20], verbose=False)
    print(f"   Scores shape: {scores_anomaly.shape}")
    print(f"   Score moyen: {scores_anomaly.mean():.4f}")
    print(f"   Score std: {scores_anomaly.std():.4f}")
    print(f"   Score min: {scores_anomaly.min():.4f}")
    print(f"   Score max: {scores_anomaly.max():.4f}")
    
    # 6. Comparaison
    print(f"\n6. Comparaison Normal vs Anomaly")
    ratio = scores_anomaly.mean() / scores_normal.mean()
    print(f"   Ratio (anomaly/normal): {ratio:.2f}x")
    
    if ratio > 2.0:
        print("    Les anomalies ont des scores plus élevés!")
    else:
        print("     Ratio faible, ajuster les hyperparamètres")
    
    print("\n" + "=" * 60)
    print(" TEST COMPLET RÉUSSI")
    print("=" * 60)
    
    return memstream, scores_normal, scores_anomaly


if __name__ == "__main__":
    # Lancer le test
    memstream, scores_normal, scores_anomaly = test_memstream()

"""
MemStream: Memory-Based Streaming Anomaly Detection
Implementation following the WWW'22 paper
"""

import numpy as np
import torch
from src.autoencoder import DenoisingAutoencoder, AETrainer
from src.memory import MemoryModule
from tqdm import tqdm

class MemStream:
    """
    MemStream: Autoencoder + Memory Module for streaming anomaly detection
    """
    def __init__(self, 
                input_dim, 
                encoding_dim=None, 
                memory_size=None,
                k_neighbors=3,
                gamma=0.0,
                threshold=1.0,
                device='cpu',
                auto_config=True):
        """
        Args:
            input_dim: dimension of input data
            encoding_dim: dimension of encodings (if None, set to 2*input_dim as in paper)
            memory_size: size of memory (N) (if None, auto-estimated)
            k_neighbors: number of neighbors (K)
            gamma: discount coefficient
            threshold: update threshold (β)
            device: 'cpu' or 'cuda'
            auto_config: automatically configure memory_size if not provided
        """
        self.input_dim = input_dim
        
        # Following paper: use D >= d (Proposition 2)
        if encoding_dim is None:
            encoding_dim = 2 * input_dim  # Paper uses 2*d
        self.encoding_dim = encoding_dim
        
        self.memory_size = memory_size  # Will be set in fit if None
        self.threshold = threshold
        self.device = device
        self.auto_config = auto_config
        
        # Autoencoder
        self.autoencoder = DenoisingAutoencoder(input_dim, encoding_dim)
        self.ae_trainer = AETrainer(self.autoencoder, device=device)
        
        # Memory Module (will be initialized in fit)
        self.memory = None
        self.k_neighbors = k_neighbors
        self.gamma = gamma
        
        # Stats for normalization
        self.mean = None
        self.std = None
        self.is_trained = False
    
    def estimate_optimal_memory_size(self, X_train, drift_speed_estimate=0.01, epsilon=0.1):
        """
        Estimate optimal memory size based on Proposition 1 from the paper.
        
        From Proposition 1: Memory size should be proportional to
        (standard deviation × √dimension × (1+ε)) / (drift speed α)
        
        Specifically: N < 2σ√(d(1+ε))/α
        
        Args:
            X_train: training data (n_samples, input_dim)
            drift_speed_estimate: estimated α (speed of distributional drift)
            epsilon: tolerance parameter (default 0.1 as suggested in paper)
        
        Returns:
            suggested_memory_size: integer
        """
        d = X_train.shape[1]  # dimension
        sigma = np.std(X_train)  # standard deviation
        
        # Proposition 1: memory should be < 2σ√(d(1+ε))/α
        theoretical_max = 2 * sigma * np.sqrt(d * (1 + epsilon)) / drift_speed_estimate
        
        print(f"   Theoretical max memory size (Proposition 1): {theoretical_max:.1f}")
        
        # Paper's practical choices (from experimental setup)
        if len(X_train) < 2000:
            # Small datasets
            candidates = [4, 8, 16, 32, 64]
        else:
            # Large datasets (intrusion detection, etc.)
            candidates = [128, 256, 512, 1024, 2048]
        
        # Choose closest to theoretical but within practical range
        optimal = min(candidates, key=lambda x: abs(x - min(theoretical_max, 2048)))
        
        # Ensure we have enough training data
        optimal = min(optimal, len(X_train) // 2)
        
        print(f"   Selected memory size: {optimal}")
        
        return optimal
    
    def fit(self, X_train, epochs=100, batch_size=32, noise_factor=0.2, 
            drift_speed_estimate=0.01, verbose=True):
        """
        Train MemStream on initial data.
        
        Following paper: Use small subset of normal data (just enough to fill memory).
        
        Args:
            X_train: numpy array (n_samples, input_dim)
            epochs: number of epochs for autoencoder training
            batch_size: batch size for training
            noise_factor: noise factor for denoising
            drift_speed_estimate: estimate of drift speed for memory sizing
            verbose: show progress
        """
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING MEMSTREAM (Following WWW'22 Paper)")
            print("=" * 60)
        
        # 1. Estimate optimal memory size if not provided
        if self.memory_size is None and self.auto_config:
            if verbose:
                print(f"\n1. Estimating optimal memory size")
            self.memory_size = self.estimate_optimal_memory_size(
                X_train, 
                drift_speed_estimate=drift_speed_estimate
            )
        elif self.memory_size is None:
            # Default fallback
            self.memory_size = min(256, len(X_train) // 2)
            if verbose:
                print(f"\n1. Using default memory size: {self.memory_size}")
        
        # 2. Use SMALL subset for training (paper: "small subset of data")
        # Following paper: use approximately memory_size samples for initialization
        n_init = min(self.memory_size * 2, len(X_train))
        X_init = X_train[:n_init]
        
        if verbose:
            print(f"\n2. Training data selection")
            print(f"   Total available: {len(X_train)} samples")
            print(f"   Using for training: {n_init} samples ({n_init/len(X_train)*100:.1f}%)")
            print(f"   Memory size: {self.memory_size}")
        
        # 3. Calculate mean/std for normalization
        if verbose:
            print(f"\n3. Computing normalization statistics")
        self.mean = np.mean(X_init, axis=0)
        self.std = np.std(X_init, axis=0) + 1e-8
        
        if verbose:
            print(f"   Mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")
            print(f"   Std range: [{self.std.min():.3f}, {self.std.max():.3f}]")
        
        # 4. Normalize
        if verbose:
            print(f"\n4. Normalizing data")
        X_normalized = (X_init - self.mean) / self.std
        
        # 5. Train autoencoder
        if verbose:
            print(f"\n5. Training autoencoder")
            print(f"   Input dim: {self.input_dim}")
            print(f"   Encoding dim: {self.encoding_dim}")
            print(f"   Epochs: {epochs}")
            print(f"   Noise factor: {noise_factor}")
        
        losses = self.ae_trainer.train(
            X_normalized,
            epochs=epochs,
            batch_size=batch_size,
            noise_factor=noise_factor,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n   Initial loss: {losses[0]:.4f}")
            print(f"   Final loss: {losses[-1]:.4f}")
            print(f"   Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")
        
        # 6. Generate encodings for memory initialization
        if verbose:
            print(f"\n6. Generating initial encodings")
        
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            encodings = self.autoencoder.encode(X_tensor).cpu().numpy()
        
        if verbose:
            print(f"   Encodings shape: {encodings.shape}")
        
        # 7. Initialize memory module
        if verbose:
            print(f"\n7. Initializing memory module")
        
        self.memory = MemoryModule(
            memory_size=self.memory_size,
            encoding_dim=self.encoding_dim,
            k_neighbors=self.k_neighbors,
            gamma=self.gamma
        )
        
        # Use memory_size encodings for initialization
        n_samples = min(len(encodings), self.memory_size)
        init_encodings = encodings[:n_samples]
        self.memory.initialize(init_encodings)
        
        self.is_trained = True
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETE")
            print("=" * 60)
            print(f"\nConfiguration:")
            print(f"  Memory size (N): {self.memory_size}")
            print(f"  K neighbors: {self.k_neighbors}")
            print(f"  Gamma (γ): {self.gamma}")
            print(f"  Threshold (β): {self.threshold}")
    
    def retrain(self, X_retrain, epochs=50, verbose=True):
        """
        Retrain the autoencoder on new data (for handling severe concept drift).
        
        Following paper Section 5.3 and Figure 4.
        
        Args:
            X_retrain: new training data
            epochs: number of retraining epochs (fewer than initial training)
            verbose: show progress
        """
        if not self.is_trained or self.memory is None:
            raise ValueError("Model must be initially trained before retraining")
        
        if verbose:
            print(f"\n{'='*50}")
            print("RETRAINING AUTOENCODER")
            print(f"{'='*50}")
            print(f"  Samples: {len(X_retrain)}")
            print(f"  Epochs: {epochs}")
        
        # Normalize with CURRENT statistics
        X_normalized = (X_retrain - self.mean) / self.std
        
        # Retrain autoencoder
        losses = self.ae_trainer.train(
            X_normalized,
            epochs=epochs,
            batch_size=32,
            noise_factor=0.2,
            verbose=verbose
        )
        
        # Regenerate memory
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            encodings = self.autoencoder.encode(X_tensor).cpu().numpy()
        
        # Reinitialize memory with new encodings
        m_size = self.memory_size if self.memory_size is not None else 256
        n_samples = min(len(encodings), m_size)
        self.memory.initialize(encodings[:n_samples])
        
        if verbose:
            print(f"✅ Retraining complete")
            print(f"   Final loss: {losses[-1]:.4f}")
    
    def score_sample(self, x):
        """
        Score a single sample.
        
        Args:
            x: numpy array (input_dim,)
        
        Returns:
            score: float (higher = more anomalous)
        """
        if not self.is_trained or self.memory is None:
            raise ValueError("Model must be trained first")
        
        # Normalize
        x_normalized = (x - self.mean) / self.std
        
        # Encode
        self.autoencoder.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_normalized).unsqueeze(0).to(self.device)
            encoding = self.autoencoder.encode(x_tensor).cpu().numpy()[0]
        
        # Compute score via memory
        score = self.memory.compute_score(encoding)
        
        # Update memory (FIFO with threshold)
        self.memory.update_fifo(encoding, score, self.threshold)
        
        return score
    
    def predict(self, X_test, verbose=True):
        """
        Predict anomaly scores for a batch (streaming simulation).
        
        Args:
            X_test: numpy array (n_samples, input_dim)
            verbose: show progress
        
        Returns:
            scores: numpy array (n_samples,)
        """
        assert self.is_trained, "Model must be trained first"
        
        scores = []
        iterator = tqdm(X_test, desc="Scoring") if verbose else X_test
        
        for x in iterator:
            score = self.score_sample(x)
            scores.append(score)
        
        return np.array(scores)
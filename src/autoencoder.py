"""
Denoising Autoencoder pour MemStream
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder avec architecture symétrique
    """
    def __init__(self, input_dim, encoding_dim):
        super(DenoisingAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Encoder: input_dim -> encoding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder: encoding_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim)
        )
    
    def forward(self, x):
        """Forward pass: x -> encoding -> reconstruction"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Retourner seulement l'encoding"""
        return self.encoder(x)


class AETrainer:
    """
    Trainer pour le Denoising Autoencoder
    """
    def __init__(self, model, learning_rate=0.01, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.training_losses = []
    
    def add_noise(self, x, noise_factor=0.2):
        """
        Ajouter du bruit gaussien (denoising)
        
        Args:
            x: tensor
            noise_factor: float, écart-type du bruit
        
        Returns:
            noisy_x: tensor avec bruit
        """
        noise = torch.randn_like(x) * noise_factor
        noisy_x = x + noise
        return noisy_x
    
    def train_batch(self, batch, noise_factor=0.2):
        """Entraîner sur un batch"""
        self.model.train()
        
        # Convertir en tensor
        x = torch.FloatTensor(batch).to(self.device)
        
        # Ajouter du bruit
        noisy_x = self.add_noise(x, noise_factor)
        
        # Forward pass
        reconstructed = self.model(noisy_x)
        loss = self.criterion(reconstructed, x)  # Reconstruire l'original
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, data_batches, noise_factor=0.2):
        """Entraîner une époque"""
        epoch_loss = 0
        
        for batch in data_batches:
            batch_loss = self.train_batch(batch, noise_factor)
            epoch_loss += batch_loss
        
        avg_loss = epoch_loss / len(data_batches)
        return avg_loss
    
    def train(self, X_train, epochs=100, batch_size=32, noise_factor=0.2, verbose=True):
        """
        Entraîner l'autoencoder
        
        Args:
            X_train: numpy array (n_samples, input_dim)
            epochs: nombre d'époques
            batch_size: taille des batchs
            noise_factor: facteur de bruit
            verbose: afficher progression
        
        Returns:
            losses: liste des pertes par époque
        """
        # Créer les batches
        n_samples = len(X_train)
        batches = [X_train[i:i+batch_size] for i in range(0, n_samples, batch_size)]
        
        # Training loop
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        
        for epoch in iterator:
            loss = self.train_epoch(batches, noise_factor)
            self.training_losses.append(loss)
            
            if verbose and epoch % 10 == 0:
                iterator.set_description(f"Epoch {epoch} - Loss: {loss:.4f}")
        
        return self.training_losses


# ============ TEST UNITAIRE ============
def test_autoencoder():
    """Test simple de l'autoencoder"""
    print("=" * 50)
    print("TEST AUTOENCODER")
    print("=" * 50)
    
    # Paramètres
    n_samples = 1000
    input_dim = 10
    encoding_dim = 5
    
    # Générer données synthétiques
    print(f"\n1. Génération de {n_samples} échantillons de dimension {input_dim}")
    X = np.random.randn(n_samples, input_dim)
    print(f"   Shape: {X.shape}")
    
    # Créer le modèle
    print(f"\n2. Création du modèle")
    print(f"   Input dim: {input_dim}")
    print(f"   Encoding dim: {encoding_dim}")
    model = DenoisingAutoencoder(input_dim, encoding_dim)
    print(f"   Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Créer le trainer
    print(f"\n3. Création du trainer")
    trainer = AETrainer(model, learning_rate=0.01)
    
    # Entraîner
    print(f"\n4. Entraînement (50 époques)")
    losses = trainer.train(X, epochs=50, batch_size=32, verbose=True)
    
    print(f"\n5. Résultats:")
    print(f"   Loss initiale: {losses[0]:.4f}")
    print(f"   Loss finale: {losses[-1]:.4f}")
    print(f"   Amélioration: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    # Test encoding
    print(f"\n6. Test encoding")
    model.eval()
    with torch.no_grad():
        test_x = torch.FloatTensor(X[:5])
        encoded = model.encode(test_x)
        decoded = model(test_x)
        
        print(f"   Input shape: {test_x.shape}")
        print(f"   Encoded shape: {encoded.shape}")
        print(f"   Decoded shape: {decoded.shape}")
        
        # Erreur de reconstruction
        mse = nn.MSELoss()(decoded, test_x).item()
        print(f"   MSE reconstruction: {mse:.4f}")
    
    print("\nTest réussi!")
    print("=" * 50)
    
    return model, trainer


if __name__ == "__main__":
    # Lancer le test
    model, trainer = test_autoencoder()


"""
Denoising Autoencoder pour MemStream
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder avec architecture symétrique
    """
    def __init__(self, input_dim, encoding_dim):
        super(DenoisingAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Encoder: input_dim -> encoding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder: encoding_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim)
        )
    
    def forward(self, x):
        """Forward pass: x -> encoding -> reconstruction"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Retourner seulement l'encoding"""
        return self.encoder(x)


class AETrainer:
    """
    Trainer pour le Denoising Autoencoder
    """
    def __init__(self, model, learning_rate=0.01, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.training_losses = []
    
    def add_noise(self, x, noise_factor=0.2):
        """
        Ajouter du bruit gaussien (denoising)
        
        Args:
            x: tensor
            noise_factor: float, écart-type du bruit
        
        Returns:
            noisy_x: tensor avec bruit
        """
        noise = torch.randn_like(x) * noise_factor
        noisy_x = x + noise
        return noisy_x
    
    def train_batch(self, batch, noise_factor=0.2):
        """Entraîner sur un batch"""
        self.model.train()
        
        # Convertir en tensor
        x = torch.FloatTensor(batch).to(self.device)
        
        # Ajouter du bruit
        noisy_x = self.add_noise(x, noise_factor)
        
        # Forward pass
        reconstructed = self.model(noisy_x)
        loss = self.criterion(reconstructed, x)  # Reconstruire l'original
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, data_batches, noise_factor=0.2):
        """Entraîner une époque"""
        epoch_loss = 0
        
        for batch in data_batches:
            batch_loss = self.train_batch(batch, noise_factor)
            epoch_loss += batch_loss
        
        avg_loss = epoch_loss / len(data_batches)
        return avg_loss
    
    def train(self, X_train, epochs=100, batch_size=32, noise_factor=0.2, verbose=True):
        """
        Entraîner l'autoencoder
        
        Args:
            X_train: numpy array (n_samples, input_dim)
            epochs: nombre d'époques
            batch_size: taille des batchs
            noise_factor: facteur de bruit
            verbose: afficher progression
        
        Returns:
            losses: liste des pertes par époque
        """
        # Créer les batches
        n_samples = len(X_train)
        batches = [X_train[i:i+batch_size] for i in range(0, n_samples, batch_size)]
        
        # Training loop
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        
        for epoch in iterator:
            loss = self.train_epoch(batches, noise_factor)
            self.training_losses.append(loss)
            
            if verbose and epoch % 10 == 0:
                iterator.set_description(f"Epoch {epoch} - Loss: {loss:.4f}")
        
        return self.training_losses


# ============ TEST UNITAIRE ============
def test_autoencoder():
    """Test simple de l'autoencoder"""
    print("=" * 50)
    print("TEST AUTOENCODER")
    print("=" * 50)
    
    # Paramètres
    n_samples = 1000
    input_dim = 10
    encoding_dim = 5
    
    # Générer données synthétiques
    print(f"\n1. Génération de {n_samples} échantillons de dimension {input_dim}")
    X = np.random.randn(n_samples, input_dim)
    print(f"   Shape: {X.shape}")
    
    # Créer le modèle
    print(f"\n2. Création du modèle")
    print(f"   Input dim: {input_dim}")
    print(f"   Encoding dim: {encoding_dim}")
    model = DenoisingAutoencoder(input_dim, encoding_dim)
    print(f"   Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Créer le trainer
    print(f"\n3. Création du trainer")
    trainer = AETrainer(model, learning_rate=0.01)
    
    # Entraîner
    print(f"\n4. Entraînement (50 époques)")
    losses = trainer.train(X, epochs=50, batch_size=32, verbose=True)
    
    print(f"\n5. Résultats:")
    print(f"   Loss initiale: {losses[0]:.4f}")
    print(f"   Loss finale: {losses[-1]:.4f}")
    print(f"   Amélioration: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    # Test encoding
    print(f"\n6. Test encoding")
    model.eval()
    with torch.no_grad():
        test_x = torch.FloatTensor(X[:5])
        encoded = model.encode(test_x)
        decoded = model(test_x)
        
        print(f"   Input shape: {test_x.shape}")
        print(f"   Encoded shape: {encoded.shape}")
        print(f"   Decoded shape: {decoded.shape}")
        
        # Erreur de reconstruction
        mse = nn.MSELoss()(decoded, test_x).item()
        print(f"   MSE reconstruction: {mse:.4f}")
    
    print("\nTest réussi!")
    print("=" * 50)
    
    return model, trainer


if __name__ == "__main__":
    # Lancer le test
    model, trainer = test_autoencoder()


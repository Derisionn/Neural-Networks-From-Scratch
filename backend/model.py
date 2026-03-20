"""
model.py
--------
Production-style neural network module.

Architecture:
    Input(2) → Dense(64) → ReLU → Dense(3) → Softmax

Weights are loaded from .npy files saved after training.
"""

import numpy as np
import os


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically-stable Softmax activation.
    Works for both single samples (1-D) and batches (2-D).
    """
    # Subtract max for numerical stability (prevents overflow)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# NeuralNetwork class
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    A two-layer fully-connected neural network that mirrors the architecture
    trained in first_network_class.ipynb.

    Expected weight files (saved with np.save after training):
        weights_dense1.npy  — shape (2, 64)
        bias_dense1.npy     — shape (1, 64)
        weights_dense2.npy  — shape (64, 3)
        bias_dense2.npy     — shape (1, 3)
    """

    # Default weight directory is the same folder as this file
    WEIGHT_FILES = {
        "W1": "weights_dense1.npy",
        "b1": "bias_dense1.npy",
        "W2": "weights_dense2.npy",
        "b2": "bias_dense2.npy",
    }

    def __init__(self, weights_dir: str = "."):
        """
        Load pre-trained weights from ``weights_dir``.

        Parameters
        ----------
        weights_dir : str
            Directory that contains the four .npy weight files.
        """
        self.weights_dir = weights_dir
        self.W1, self.b1, self.W2, self.b2 = self._load_weights()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_weights(self):
        """Load and return all four weight arrays."""
        loaded = {}
        for key, filename in self.WEIGHT_FILES.items():
            path = os.path.join(self.weights_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Weight file not found: '{path}'. "
                    "Make sure you have saved your trained weights with np.save()."
                )
            loaded[key] = np.load(path)

        return loaded["W1"], loaded["b1"], loaded["W2"], loaded["b2"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Run a forward pass and return raw Softmax probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, 2) or (2,) for a single sample.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_samples, 3) or (3,).
        """
        # Ensure 2-D input so matrix multiply works for both single + batch
        single_sample = X.ndim == 1
        if single_sample:
            X = X[np.newaxis, :]  # (1, 2)

        # Layer 1: Dense → ReLU
        z1 = X @ self.W1 + self.b1          # (n, 64)
        a1 = relu(z1)                        # (n, 64)

        # Layer 2: Dense → Softmax
        z2 = a1 @ self.W2 + self.b2          # (n, 3)
        probs = softmax(z2)                  # (n, 3)

        return probs[0] if single_sample else probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted class index (argmax of Softmax output).

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, 2) or (2,).

        Returns
        -------
        int or np.ndarray
            Predicted class index / indices.
        """
        probs = self.forward(X)
        return int(np.argmax(probs, axis=-1)) if probs.ndim == 1 else np.argmax(probs, axis=-1)

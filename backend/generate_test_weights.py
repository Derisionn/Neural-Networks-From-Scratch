"""
generate_test_weights.py
------------------------
Generates random .npy weight files so you can test the API immediately
WITHOUT re-running the full training loop.

These weights produce random (untrained) predictions — swap them out with
real weights saved from first_network_class.ipynb when you're ready.

Run once:
    python generate_test_weights.py
"""

import numpy as np

np.random.seed(42)

# Dense layer 1: input(2) → 64 neurons
W1 = 0.01 * np.random.randn(2, 64).astype(np.float32)
b1 = np.zeros((1, 64), dtype=np.float32)

# Dense layer 2: 64 → 3 classes
W2 = 0.01 * np.random.randn(64, 3).astype(np.float32)
b2 = np.zeros((1, 3), dtype=np.float32)

np.save("weights_dense1.npy", W1)
np.save("bias_dense1.npy",    b1)
np.save("weights_dense2.npy", W2)
np.save("bias_dense2.npy",    b2)

print("✅ Test weight files created:")
print(f"   weights_dense1.npy  shape={W1.shape}")
print(f"   bias_dense1.npy     shape={b1.shape}")
print(f"   weights_dense2.npy  shape={W2.shape}")
print(f"   bias_dense2.npy     shape={b2.shape}")
print("\n🚀 You can now start the server:  uvicorn main:app --reload")

"""
save_weights.py
---------------
Run this script ONCE after training in first_network_class.ipynb to export
the trained weights as .npy files that the API will load.

Usage (in the notebook or a terminal):
    python save_weights.py

Alternatively, paste the np.save() calls directly at the end of your
training loop in the notebook.
"""

import numpy as np

# ── Paste your trained layer objects here ────────────────────────────────────
# These names must match the variable names used in first_network_class.ipynb
# e.g.  dense_1, dense_2

# np.save("weights_dense1.npy", dense_1.weights)
# np.save("bias_dense1.npy",    dense_1.biases)
# np.save("weights_dense2.npy", dense_2.weights)
# np.save("bias_dense2.npy",    dense_2.biases)

# ── Quick sanity-check (run after un-commenting the lines above) ─────────────
if __name__ == "__main__":
    import os
    files = [
        "weights_dense1.npy",
        "bias_dense1.npy",
        "weights_dense2.npy",
        "bias_dense2.npy",
    ]
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        print("⚠️  The following weight files are MISSING:")
        for f in missing:
            print(f"    {f}")
        print("\nUn-comment the np.save() lines above and run this script again.")
    else:
        print("✅ All weight files found:")
        for f in files:
            arr = np.load(f)
            print(f"    {f:30s}  shape={arr.shape}")

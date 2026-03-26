import numpy as np
import os

# ── Spiral Data Generator ──────────────────────────────────────────────────
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# ── Layer and Activation Classes ───────────────────────────────────────────
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.backward_outputs = None
    def forward(self, inputs, y_true):
        # Softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        # Loss
        y_pred_clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_true)), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return np.mean(-np.log(correct_confidences))
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

# ── Training Loop ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = spiral_data(points=100, classes=3)
    
    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD(learning_rate=1.0)

    print("🚀 Starting training...")
    for epoch in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)
        
        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)
        
        if not epoch % 1000:
            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')
            
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

    # ── Save Weights ────────────────────────────────────────────────────────
    print("💾 Saving weights to backend/ directory...")
    # Ensure we are in the right directory or use relative paths
    output_dir = 'backend'
    if not os.path.exists(output_dir):
        # Fallback if run from inside backend
        output_dir = '.'
        
    np.save(os.path.join(output_dir, "weights_dense1.npy"), dense1.weights)
    np.save(os.path.join(output_dir, "bias_dense1.npy"),    dense1.biases)
    np.save(os.path.join(output_dir, "weights_dense2.npy"), dense2.weights)
    np.save(os.path.join(output_dir, "bias_dense2.npy"),    dense2.biases)
    print("✅ Done!")

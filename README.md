# 🧠 Neural Networks From Scratch

> A full-stack machine learning project demonstrating deep understanding of neural network internals — no PyTorch, no TensorFlow. Just NumPy, math, and a production-grade API.

---

## 🔍 Overview

This project implements a **multi-class neural network classifier from the ground up** using only NumPy. The model is trained on the classic **Spiral Dataset** — a non-linearly separable 3-class problem — and is served as a live REST API with an interactive React frontend.

Rather than relying on ML frameworks, every component is hand-crafted:

- **Forward propagation** through Dense layers
- **Backpropagation** with analytically-derived gradients
- **SGD optimizer** updating weights and biases
- **Softmax + Categorical Cross-Entropy** fused for numerical stability

---

## 🏗️ Architecture

```
Input(2) ──► Dense(64) ──► ReLU ──► Dense(3) ──► Softmax ──► Predicted Class
```

| Layer       | Shape      | Activation |
|-------------|------------|------------|
| Input        | `(n, 2)`   | —          |
| Hidden (Dense 1) | `(n, 64)` | ReLU   |
| Output (Dense 2) | `(n, 3)`  | Softmax |

Trained for **10,000 epochs** on 300 samples (100 per class) using vanilla SGD with a learning rate of `1.0`.

---

## 🚀 Live Demo

| Service  | Link |
|----------|------|
| 🌐 Frontend | [neural-networks-from-scratch.vercel.app](https://neural-networks-from-scratch.vercel.app) |
| ⚙️ API Docs  | [neural-network-api.onrender.com/docs](https://neural-network-api-bhyn.onrender.com/docs) |

---

## 📁 Project Structure

```
Neural-Networks-From-Scratch/
│
├── backpropagation_single_neuron.ipynb   # Step-by-step backprop derivation
├── first_network_class.ipynb             # Full training loop + spiral visualization
│
├── backend/                              # FastAPI REST API
│   ├── model.py                          # NeuralNetwork inference class (NumPy)
│   ├── train_and_save.py                 # Training script — saves .npy weights
│   ├── main.py                           # FastAPI app with /predict endpoints
│   ├── requirements.txt
│   └── *.npy                             # Saved weight files
│
├── frontend/                             # React + Vite UI
│   └── src/
│       └── App.jsx                       # Interactive classifier UI + Recharts
│
└── render.yaml                           # One-click Render deployment config
```

---

## 🔬 Notebooks

### `backpropagation_single_neuron.ipynb`
A ground-up walkthrough of the backpropagation algorithm on a single neuron — deriving gradients by hand and verifying them numerically.

### `first_network_class.ipynb`
Full training pipeline on the Spiral Dataset:
- Spiral data generation
- Modular `Layer_Dense` and `Activation_ReLU` classes
- Fused `Softmax + CategoricalCrossEntropy` for stable gradient computation
- Training loop with live accuracy/loss logging
- Weight export via `np.save()`

---

## ⚙️ Backend — FastAPI

The trained weights are loaded once at startup and served via a production-style REST API.

### Endpoints

| Method | Endpoint          | Description                  |
|--------|-------------------|------------------------------|
| `GET`  | `/`               | Welcome message              |
| `GET`  | `/health`         | Model readiness health check |
| `POST` | `/predict`        | Single-sample prediction     |
| `POST` | `/predict_batch`  | Batch prediction             |

### Example Request

```bash
curl -X POST https://neural-network-api-bhyn.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -0.3]}'
```

### Example Response

```json
{
  "predicted_class": 2,
  "probabilities": [0.032, 0.118, 0.850]
}
```

---

## 🖥️ Frontend — React + Vite

An interactive React UI that lets users input 2D coordinates and visualize the model's predictions in real time.

**Features:**
- Live API calls to the Render backend
- Recharts bar chart showing per-class probabilities
- Animated probability progress bars
- Dark mode glassmorphism design

**Tech Stack:** React 19 · Vite · Recharts · Vanilla CSS

---

## 🛠️ Local Setup

### Backend

```bash
cd backend
pip install -r requirements.txt

# (Optional) Retrain the model and save weights
python train_and_save.py

# Start the API server
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:5173`.

> Make sure to update `API_URL` in `src/App.jsx` to point to `http://localhost:8000` for local development.

---

## ☁️ Deployment

| Service  | Platform | Config |
|----------|----------|--------|
| Backend  | [Render](https://render.com) | `render.yaml` (auto-detected) |
| Frontend | [Vercel](https://vercel.com) | `frontend/vercel.json` |

The `render.yaml` at the root enables one-click backend deployment — push to GitHub and Render handles the rest.

---

## 💡 Key Concepts Demonstrated

- Manual **forward pass** and **backward pass** implementation
- **Chain rule** applied across layers for gradient computation
- **Numerically-stable Softmax** (subtract max trick)
- **Fused Softmax-Loss backward pass** for clean gradient derivation
- Weight persistence and loading with **NumPy `.npy` files**
- **Production API design** with FastAPI: Pydantic validation, lifespan events, CORS, health checks
- **Full-stack deployment**: Render (backend) + Vercel (frontend)

---

## 🧰 Tech Stack

| Layer       | Technology |
|-------------|------------|
| ML Engine   | Python · NumPy |
| API         | FastAPI · Uvicorn · Pydantic |
| Frontend    | React 19 · Vite · Recharts |
| Deployment  | Render · Vercel |
| Notebooks   | Jupyter |

---


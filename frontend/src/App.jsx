import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, CartesianGrid,
} from 'recharts'
import './App.css'

// ── Config ─────────────────────────────────────────────────────────────────
// Set your Render backend URL in frontend/.env  →  VITE_API_URL=https://...
const API_URL = import.meta.env.VITE_API_URL

const CLASS_LABELS = ['Class 0', 'Class 1', 'Class 2']
const CLASS_COLORS = ['#6366f1', '#8b5cf6', '#a78bfa']

// ── Custom Tooltip ──────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip">
        <p className="tooltip-label">{label}</p>
        <p className="tooltip-value">{(payload[0].value * 100).toFixed(2)}%</p>
      </div>
    )
  }
  return null
}

// ── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [feature1, setFeature1] = useState('')
  const [feature2, setFeature2] = useState('')
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  const isValid = feature1 !== '' && feature2 !== '' && !isNaN(feature1) && !isNaN(feature2)

  async function handlePredict() {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ features: [parseFloat(feature1), parseFloat(feature2)] }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Server error (${res.status})`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Build chart data from probabilities
  const chartData = result
    ? result.probabilities.map((p, i) => ({ name: CLASS_LABELS[i], probability: p }))
    : []

  return (
    <div className="page">
      {/* ── Header ── */}
      <header className="header">
        <span className="header-icon">🧠</span>
        <h1>Neural Network Predictor</h1>
        <p>Spiral Dataset Classifier — trained from scratch with NumPy</p>
      </header>

      {/* ── Card ── */}
      <main className="card">

        {/* Inputs */}
        <section className="inputs-section">
          <div className="input-group">
            <label htmlFor="f1">Feature 1 &mdash; x</label>
            <input
              id="f1"
              type="number"
              placeholder="e.g. 0.5"
              value={feature1}
              onChange={(e) => setFeature1(e.target.value)}
              disabled={loading}
            />
          </div>
          <div className="input-group">
            <label htmlFor="f2">Feature 2 &mdash; y</label>
            <input
              id="f2"
              type="number"
              placeholder="e.g. -0.3"
              value={feature2}
              onChange={(e) => setFeature2(e.target.value)}
              disabled={loading}
            />
          </div>
        </section>

        {/* Button */}
        <button
          className={`predict-btn ${loading ? 'loading' : ''}`}
          onClick={handlePredict}
          disabled={!isValid || loading}
        >
          {loading ? <span className="spinner" /> : null}
          {loading ? 'Predicting…' : 'Predict'}
        </button>

        {/* Error */}
        {error && (
          <div className="error-box">
            <span>⚠️</span> {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <section className="results">

            {/* Predicted class badge */}
            <div className="class-badge">
              <span className="class-label">Predicted Class</span>
              <span
                className="class-value"
                style={{ color: CLASS_COLORS[result.predicted_class] }}
              >
                {result.predicted_class}
              </span>
              <span className="class-name" style={{ color: CLASS_COLORS[result.predicted_class] }}>
                {CLASS_LABELS[result.predicted_class]}
              </span>
            </div>

            {/* Bar chart */}
            <div className="chart-wrapper">
              <h3 className="chart-title">Class Probabilities</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 13 }} />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                  <Bar dataKey="probability" radius={[6, 6, 0, 0]} maxBarSize={60}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill={CLASS_COLORS[i]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Probability rows */}
            <div className="prob-list">
              {result.probabilities.map((p, i) => (
                <div key={i} className="prob-row">
                  <span className="prob-label" style={{ color: CLASS_COLORS[i] }}>
                    {CLASS_LABELS[i]}
                  </span>
                  <div className="prob-bar-bg">
                    <div
                      className="prob-bar-fill"
                      style={{ width: `${(p * 100).toFixed(2)}%`, background: CLASS_COLORS[i] }}
                    />
                  </div>
                  <span className="prob-pct">{(p * 100).toFixed(2)}%</span>
                </div>
              ))}
            </div>

          </section>
        )}
      </main>

      <footer className="footer">Built with React + FastAPI + NumPy</footer>
    </div>
  )
}

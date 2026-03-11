"""
examples/quickstart.py — Getting started with pycheron in 5 lines.

Run:
    python examples/quickstart.py
"""

import numpy as np
import pandas as pd
import pycheron

# ── 1. Create (or load) your data ────────────────────────────────────────────
rng = np.random.RandomState(42)
n = 500
df = pd.DataFrame({
    "age":        rng.randint(18, 80, n).astype(float),
    "income":     rng.exponential(50_000, n),
    "gender":     rng.choice(["M", "F"], n),
    "education":  rng.choice(["high school", "bachelor", "master", "phd"], n),
    "experience": rng.randint(0, 40, n),
    "purchased":  rng.choice([0, 1], n, p=[0.6, 0.4]),
})

# ── 2. Train — one line ───────────────────────────────────────────────────────
model = pycheron.train(df, target="purchased")
# Output: progress, CV score, final evaluation report

# ── 3. Predict ────────────────────────────────────────────────────────────────
new_customers = df.drop(columns=["purchased"]).head(10)
predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)
print("Predictions:", predictions)
print("Probabilities (class 1):", probabilities[:, 1].round(3))

# ── 4. Explain ────────────────────────────────────────────────────────────────
try:
    explanation = model.explain(new_customers)
    print("\nTop features:")
    print(explanation.summary(top_n=5))
except ImportError:
    print("Install shap for explanations: pip install shap")

# ── 5. Save & reload ──────────────────────────────────────────────────────────
model.save("./my_model")
loaded = pycheron.load_model("./my_model")
print("\nLoaded model:", loaded)

# ── 6. AutoML (optional) ──────────────────────────────────────────────────────
print("\n--- AutoML ---")
best_model = pycheron.auto_train(df, target="purchased", time_budget=60)
best_model.leaderboard()

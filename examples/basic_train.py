import pymlkit
import pandas as pd
import numpy as np

# Generate synthetic dataset
rng = np.random.RandomState(42)

df = pd.DataFrame({
    "age": rng.randint(18, 60, 200),
    "salary": rng.randint(20000, 100000, 200),
    "department": rng.choice(["tech", "sales", "hr"], 200),
    "target": rng.choice([0, 1], 200)
})

print("Dataset shape:", df.shape)

# Train model
model = pymlkit.train(
    df,
    target="target",
    verbose=2
)

# Evaluate model
results = model.evaluate()

print("Evaluation:", results)
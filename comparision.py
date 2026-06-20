from train_xgboost import train_xgb
from train_sarima import train_sarima
from train_lstm import train_lstm

print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

results = {}

# XGBoost
try:
    results["XGBoost"] = train_xgb()
except Exception as e:
    print(f"XGBoost failed: {e}")

# SARIMA
try:
    results["SARIMA"] = train_sarima()
except Exception as e:
    print(f"SARIMA failed: {e}")

# LSTM
try:
    results["LSTM"] = train_lstm()
except Exception as e:
    print(f"LSTM failed: {e}")


if len(results) == 0:
    raise ValueError("No models were successfully trained.")

# Sort models by MAE
sorted_results = sorted(
    results.items(),
    key=lambda x: x[1]
)

print("\n")
print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)

for model, score in sorted_results:
    print(f"{model:<15} MAE = {score:,.2f}")

# Best model
best_model = min(
    results,
    key=results.get
)

best_score = results[best_model]

print("\n" + "=" * 50)
print(f"BEST MODEL : {best_model}")
print(f"BEST MAE   : {best_score:,.2f}")
print("=" * 50)
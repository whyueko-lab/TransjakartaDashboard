import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1️⃣ Load & Preprocessing
# -------------------------
df = pd.read_csv("dfTransjakarta.csv")
df["tapInTime"] = pd.to_datetime(df["tapInTime"])
df["hour"] = df["tapInTime"].dt.hour
df["weekday"] = df["tapInTime"].dt.weekday  # 0=Senin, 6=Minggu

# Buat target: jumlah penumpang per corridor per jam
df_grouped = (
    df.groupby(["corridorName", "hour", "weekday"])
    .size()
    .reset_index(name="passengers")
)

# Drop rows with null corridor
df_grouped = df_grouped.dropna(subset=["corridorName"])

# -------------------------
# 2️⃣ Encode Categorical Feature
# -------------------------
df_grouped = pd.get_dummies(df_grouped, columns=["corridorName"], drop_first=True)

# -------------------------
# 3️⃣ Split Data
# -------------------------
X = df_grouped.drop("passengers", axis=1)
y = df_grouped["passengers"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 4️⃣ Train Random Forest
# -------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------
# 5️⃣ Evaluate Model
# -------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {mse**0.5:.2f}")
print(f"R2 Score: {r2:.2f}")

# -------------------------
# 6️⃣ Visualisasi Prediksi vs Aktual
# -------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Passengers")
plt.ylabel("Predicted Passengers")
plt.title("Prediksi vs Aktual Jumlah Penumpang per Corridor per Jam")
plt.plot([0, max(y_test)], [0, max(y_test)], "r--")  # garis ideal
plt.tight_layout()
plt.show()

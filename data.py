import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# -------------------------
# 1️⃣ Load Data
# -------------------------
df = pd.read_csv("dfTransjakarta.csv")

# Convert time columns to datetime
df["tapInTime"] = pd.to_datetime(df["tapInTime"])
df["tapOutTime"] = pd.to_datetime(df["tapOutTime"], errors="coerce")

# -------------------------
# 2️⃣ Tren Jumlah Penumpang per Jam
# -------------------------
df_trend = df.copy()
df_trend["hour"] = df_trend["tapInTime"].dt.hour
passengers_per_hour = df_trend.groupby("hour").size()

plt.figure(figsize=(10, 5))
sns.barplot(x=passengers_per_hour.index, y=passengers_per_hour.values, color="skyblue")
plt.title("Tren Jumlah Penumpang Transjakarta per Jam")
plt.xlabel("Jam")
plt.ylabel("Jumlah Penumpang")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

# -------------------------
# 3️⃣ Heatmap Tap-In & Tap-Out
# -------------------------
heat_data_in = df[["tapInStopsLat", "tapInStopsLon"]].dropna().values.tolist()
heat_data_out = df[["tapOutStopsLat", "tapOutStopsLon"]].dropna().values.tolist()

# Map center: Jakarta
m = folium.Map(location=[-6.2, 106.8], zoom_start=12)

# Add tap-in heatmap
HeatMap(
    heat_data_in,
    radius=8,
    blur=10,
    gradient={0.4: "blue", 0.65: "lime", 1: "red"},
    name="Tap-In",
).add_to(m)
# Add tap-out heatmap
HeatMap(
    heat_data_out,
    radius=8,
    blur=10,
    gradient={0.4: "purple", 0.65: "orange", 1: "red"},
    name="Tap-Out",
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save map
m.save("heatmap_transjakarta.html")
print("Heatmap saved as heatmap_transjakarta.html")

# -------------------------
# 4️⃣ Distribusi Penumpang per Corridor
# -------------------------
df_corridor = df.groupby("corridorName").size().sort_values(ascending=False)

top_corridor = df_corridor.head(30)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_corridor.index, y=top_corridor.values, palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.title("Top 30 Corridor dengan Jumlah Penumpang Terbanyak")
plt.ylabel("Jumlah Penumpang")
plt.xlabel("Corridor")

# Tambahkan qty di ujung bar
for index, value in enumerate(top_corridor.values):
    plt.text(index, value + -50, str(value), ha="center")

plt.tight_layout()
plt.show()


# -------------------------
# 5️⃣ Distribusi Pembayaran
# -------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["payAmount"].dropna(), bins=20, color="orange")
plt.title("Distribusi Pembayaran Transjakarta")
plt.xlabel("Jumlah Pembayaran (Rp)")
plt.ylabel("Frekuensi")
plt.tight_layout()
plt.show()

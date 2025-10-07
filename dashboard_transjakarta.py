# dashboard_transjakarta_full.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import folium_static


# -------------------------
# 1️⃣ Load & Preprocess
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dfTransjakarta.csv")
    df["tapInTime"] = pd.to_datetime(df["tapInTime"])
    df["hour"] = df["tapInTime"].dt.hour
    df["weekday"] = df["tapInTime"].dt.weekday  # 0=Senin
    df_grouped = (
        df.groupby(["corridorName", "hour", "weekday"])
        .size()
        .reset_index(name="passengers")
    )
    df_grouped = df_grouped.dropna(subset=["corridorName"])
    return df, df_grouped


df, df_grouped = load_data()

# -------------------------
# 2️⃣ Train Random Forest
# -------------------------
df_encoded = pd.get_dummies(df_grouped, columns=["corridorName"], drop_first=True)
X = df_encoded.drop("passengers", axis=1)
y = df_encoded["passengers"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------
# 3️⃣ Streamlit UI
# -------------------------
st.title("Dashboard Transjakarta - Analitik & Prediksi Penumpang")

# Sidebar input
st.sidebar.header("Filter Input Prediksi")
hour_input = st.sidebar.slider("Pilih Jam (0-23)", 0, 23, 8)
weekday_input = st.sidebar.selectbox(
    "Pilih Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
)
weekday_map = {
    "Senin": 0,
    "Selasa": 1,
    "Rabu": 2,
    "Kamis": 3,
    "Jumat": 4,
    "Sabtu": 5,
    "Minggu": 6,
}
weekday_val = weekday_map[weekday_input]
corridor_input = st.sidebar.selectbox(
    "Pilih Corridor", df_grouped["corridorName"].unique()
)

# Input untuk prediksi
input_dict = {"hour": [hour_input], "weekday": [weekday_val]}
for col in X.columns:
    if col.startswith("corridorName_"):
        input_dict[col] = [1 if col == f"corridorName_{corridor_input}" else 0]
input_df = pd.DataFrame(input_dict)
pred_passengers = model.predict(input_df)[0]

st.subheader(f"Prediksi Jumlah Penumpang: {int(pred_passengers)} orang")

# -------------------------
# 4️⃣ Visualisasi Tren Aktual
# -------------------------
st.subheader("Tren Jumlah Penumpang Aktual per Jam")
corridor_actual = df_grouped[df_grouped["corridorName"] == corridor_input]
plt.figure(figsize=(10, 4))
sns.lineplot(data=corridor_actual, x="hour", y="passengers", marker="o")
plt.axvline(hour_input, color="red", linestyle="--", label="Jam Dipilih")
plt.title(f"Tren Jumlah Penumpang Corridor: {corridor_input}")
plt.xlabel("Jam")
plt.ylabel("Jumlah Penumpang")
plt.legend()
st.pyplot(plt.gcf())

# -------------------------
# 5️⃣ Top Corridor Bar Chart
# -------------------------
st.subheader("Top 10 Corridor dengan Jumlah Penumpang Tertinggi")
corridor_sum = (
    df_grouped.groupby("corridorName")["passengers"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
plt.figure(figsize=(10, 5))
sns.barplot(x=corridor_sum.values, y=corridor_sum.index, palette="viridis")
for index, value in enumerate(corridor_sum.values):
    plt.text(value + 5, index, str(value), va="center")
plt.xlabel("Jumlah Penumpang")
plt.ylabel("Corridor")
plt.tight_layout()
st.pyplot(plt.gcf())

# -------------------------
# 6️⃣ Heatmap Geospasial
# -------------------------
st.subheader("Heatmap Tap-In Transjakarta")
df_map = df.dropna(subset=["tapInStopsLat", "tapInStopsLon"])
m = folium.Map(location=[-6.2, 106.8], zoom_start=11)
for _, row in df_map.iterrows():
    folium.CircleMarker(
        location=[row["tapInStopsLat"], row["tapInStopsLon"]],
        radius=2,
        color="blue",
        fill=True,
        fill_opacity=0.5,
    ).add_to(m)
folium_static(m)

st.markdown("---")
st.markdown("Data dan model disiapkan oleh Wahyu Eko Suroso - Copyright 2025")
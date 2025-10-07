# dashboard_transjakarta.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1️⃣ Load & Preprocessing
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dfTransjakarta.csv")
    df['tapInTime'] = pd.to_datetime(df['tapInTime'])
    df['hour'] = df['tapInTime'].dt.hour
    df['weekday'] = df['tapInTime'].dt.weekday  # 0=Senin
    df_grouped = df.groupby(['corridorName', 'hour', 'weekday']).size().reset_index(name='passengers')
    df_grouped = df_grouped.dropna(subset=['corridorName'])
    return df_grouped

df_grouped = load_data()

# -------------------------
# 2️⃣ Encode Corridor
# -------------------------
df_encoded = pd.get_dummies(df_grouped, columns=['corridorName'], drop_first=True)
X = df_encoded.drop('passengers', axis=1)
y = df_encoded['passengers']

# -------------------------
# 3️⃣ Train Random Forest
# -------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------
# 4️⃣ Streamlit UI
# -------------------------
st.title("Prediksi Jumlah Penumpang Transjakarta per Corridor & Jam")

# Sidebar input
hour_input = st.sidebar.slider("Pilih Jam (0-23)", 0, 23, 8)
weekday_input = st.sidebar.selectbox("Pilih Hari", ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"])
weekday_map = {"Senin":0,"Selasa":1,"Rabu":2,"Kamis":3,"Jumat":4,"Sabtu":5,"Minggu":6}
weekday_val = weekday_map[weekday_input]

corridor_input = st.sidebar.selectbox("Pilih Corridor", df_grouped['corridorName'].unique())

# Buat dataframe input untuk prediksi
input_dict = {'hour':[hour_input], 'weekday':[weekday_val]}
for col in X.columns:
    if col.startswith('corridorName_'):
        input_dict[col] = [1 if col == f"corridorName_{corridor_input}" else 0]

input_df = pd.DataFrame(input_dict)

# Prediksi
pred_passengers = model.predict(input_df)[0]
st.subheader(f"Prediksi Jumlah Penumpang: {int(pred_passengers)} orang")

# -------------------------
# 5️⃣ Visualisasi Tren Prediksi vs Aktual
# -------------------------
st.subheader("Grafik Tren Jumlah Penumpang per Jam (Actual)")
corridor_actual = df_grouped[df_grouped['corridorName']==corridor_input]
plt.figure(figsize=(10,4))
sns.lineplot(data=corridor_actual, x='hour', y='passengers', marker='o')
plt.axvline(hour_input, color='red', linestyle='--', label="Jam Dipilih")
plt.title(f"Tren Jumlah Penumpang Corridor: {corridor_input}")
plt.xlabel("Jam")
plt.ylabel("Jumlah Penumpang")
plt.legend()
st.pyplot(plt.gcf())
